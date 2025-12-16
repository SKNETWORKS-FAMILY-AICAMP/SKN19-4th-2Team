# chat/views.py

import json
import time
import concurrent.futures  # [추가] 비동기 작업을 위한 모듈
from django.shortcuts import render, get_object_or_404, redirect
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Max

# [주의] login_required 제거함 (비회원 접근 허용을 위해)
from .models import ChatHistory, Chat

# LLM 모듈
from llm_module.main import get_graph_agent
from llm_module.SYSTEM_PROMPT import SYSTEM_PROMPT
from llm_module.memory_utils import convert_db_chats_to_langchain
from openai import OpenAI
from django.conf import settings

agent_executor = get_graph_agent()

client = OpenAI(api_key=getattr(settings, "OPENAI_API_KEY", None))


def generate_history_title_by_llm(first_message: str) -> str:
    """
    첫 사용자 메시지를 기반으로 채팅방 제목(20자 이내)을 생성한다.
    """
    try:
        prompt = (
            "다음 사용자의 첫 질문을 보고, 채팅방 제목으로 쓸 짧은 한글 문구를 만들어줘. "
            "20자 이내로, 마침표 없이 간단하게.\n\n"
            f"질문: {first_message}"
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",  # 필요하면 모델 이름 바꿔도 됨
            messages=[
                {"role": "system", "content": "너는 채팅방 제목을 짧게 요약해주는 도우미야."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=50,
            temperature=0.3,
        )
        title = resp.choices[0].message.content.strip()

        # 길면 잘라주기
        if len(title) > 20:
            title = title[:20]

        # 혹시 비어 있으면 fallback
        if not title:
            title = first_message[:20] + "..."

        return title

    except Exception:
        # LLM 실패해도 앱이 안 죽도록 안전장치
        return first_message[:20] + "..."


# =========================================================
# [핵심] 현재 사용자의 히스토리를 가져오는 함수 (회원/비회원 분기)
# =========================================================
def get_current_history(request):
    # 1. 로그인한 회원인 경우
    if request.user.is_authenticated:
        history = (
            ChatHistory.objects.filter(user=request.user)
            .order_by("-created_at")
            .first()
        )
        if not history:
            history = ChatHistory.objects.create(
                user=request.user, order_num=1, description="새로운 대화"
            )
        return history

    # 2. 비회원(Guest)인 경우 -> 세션 ID 사용
    else:
        # 세션 키가 없으면 생성
        if not request.session.session_key:
            request.session.save()

        session_id = request.session.session_key

        # 세션 ID로 조회 (user는 Null인 것만)
        history = (
            ChatHistory.objects.filter(session_id=session_id, user__isnull=True)
            .order_by("-created_at")
            .first()
        )

        if not history:
            history = ChatHistory.objects.create(
                user=None,  # 비회원이므로 Null
                session_id=session_id,
                order_num=1,
                description="게스트 대화",
            )
        return history


# =========================================================
# 뷰: 채팅 화면
# =========================================================
def chat_interface(request):
    """
    전체 채팅 페이지 렌더링
    """
    user = request.user
    selected_history = None

    # 1. 채팅 목록 가져오기 (정렬 기준 변경: created_at -> order_num)
    if user.is_authenticated:
        # [수정] order_num 내림차순 정렬 (높은 번호가 위로)
        history_list = ChatHistory.objects.filter(user=user).order_by("-order_num")
    else:
        # 비회원 세션 처리
        if not request.session.session_key:
            request.session.save()
        session_id = request.session.session_key

        # [수정] order_num 내림차순 정렬
        history_list = ChatHistory.objects.filter(
            session_id=session_id, user__isnull=True
        ).order_by("-order_num")

    # 2. 특정 채팅방 선택 로직 (URL 파라미터 ?history_id=123)
    target_id = request.GET.get("history_id")

    if target_id:
        selected_history = history_list.filter(history_id=target_id).first()

    # 3. 선택된 게 없으면 -> 목록의 첫 번째(가장 높은 번호) 선택 or 새로 생성
    if not selected_history:
        if history_list.exists():
            selected_history = history_list.first()
        else:
            # 기록이 없으면 새 방 생성 (1번방)
            if user.is_authenticated:
                selected_history = ChatHistory.objects.create(
                    user=user, order_num=1, description="새로운 대화"
                )
            else:
                session_id = request.session.session_key
                selected_history = ChatHistory.objects.create(
                    session_id=session_id,
                    user=None,
                    order_num=1,
                    description="게스트 대화",
                )

            # (참고) 방금 만든 방은 쿼리셋 재평가 시 자동으로 반영됨

    # 4. 선택된 방의 대화 내용 가져오기 (대화 내용은 순서대로 1,2,3...)
    chats = Chat.objects.filter(history=selected_history).order_by("order_num")

    context = {
        "user_id": user.id if user.is_authenticated else "guest",
        "selected_history_id": selected_history.history_id,
        "chat_history": chats,
        "history_list": history_list,
    }

    return render(request, "chat/chat_interface.html", context)


# =========================================================
# API: 채팅 스트리밍 (비동기 제목 생성 적용)
# =========================================================
@csrf_exempt
def chat_stream_api(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_input = data.get("message", "")
            history_id = data.get("history_id")
        except:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        if not user_input or not history_id:
            return JsonResponse({"error": "Missing data"}, status=400)

        # 1. 히스토리 객체 가져오기 (회원/비회원 분기)
        if request.user.is_authenticated:
            history = get_object_or_404(
                ChatHistory, history_id=history_id, user=request.user
            )
        else:
            if not request.session.session_key:
                return JsonResponse({"error": "Session expired"}, status=403)
            history = get_object_or_404(
                ChatHistory,
                history_id=history_id,
                session_id=request.session.session_key,
            )

        # ------------------------------------------------------------------
        # [순서 관리]
        # ------------------------------------------------------------------
        last_order = history.chats.aggregate(Max("order_num"))["order_num__max"] or 0
        current_save_order = last_order + 1

        # 👉 첫 메시지인지 여부 체크
        is_first_message = (last_order == 0)

        # 2. [사용자 메시지 저장]
        user_chat = Chat.objects.create(
            history=history,
            type="HUMAN",
            content=user_input,
            order_num=current_save_order,
        )

        current_save_order += 1

        # =====================================================
        # [최적화] 제목 생성은 여기서 기다리지 않고(Block X),
        # 아래 event_stream 내부의 별도 스레드(Thread)에게 맡깁니다.
        # =====================================================

        # 3. LangChain 메시지 변환
        if request.user.is_authenticated:
            user_nickname = request.user.first_name or "사용자"
        else:
            user_nickname = "게스트"

        dynamic_system_prompt = SYSTEM_PROMPT + f"""

------------------------------------
[대화/호칭 관련 추가 지침]
------------------------------------
- 너의 이름은 "Pai" 이다. (Patent AI 의 줄임말)
  필요할 때 "저는 특허 AI 어시스턴트 Pai입니다."처럼 자신을 소개해도 된다.
- 현재 사용자의 닉네임(표시 이름)은 "{user_nickname}" 이다.
- 답변할 때는 존댓말을 사용하고,
  너무 과하게 반복하지 않는 선에서 자연스럽게 "{user_nickname}님"이라고 불러 준다.
- 단, 매 문장마다 부르는 것은 피하고, 필요할 때 한두 번 정도만 사용한다.
"""

        db_chats = Chat.objects.filter(history=history).order_by("order_num")
        langchain_messages = convert_db_chats_to_langchain(
            db_chats,
            system_prompt=dynamic_system_prompt,
        )

        config = {"configurable": {"thread_id": str(history.history_id)}}

        def event_stream():
            nonlocal current_save_order

            full_ai_response = ""
            seen_tool_ids = set()

            # [수정] DB 객체를 미리 잡아두기 위한 변수
            ai_message_obj = None

            last_save_time = time.time()

            # --------------------------------------------------------
            # [비동기] 제목 생성 작업자 준비
            # --------------------------------------------------------
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            title_future = None
            title_sent = False  # 클라이언트에 보냈는지 체크

            # (내부 함수) 제목 생성 및 DB 저장 작업
            def title_task():
                generated_title = generate_history_title_by_llm(user_input)
                # DB 저장도 스레드 안에서 처리
                history.description = generated_title
                history.save(update_fields=["description"])
                return generated_title

            try:
                # 1. 첫 메시지라면, 제목 생성 '숙제'를 백그라운드 스레드에 던져놓고 바로 다음 줄로 진행!
                if is_first_message:
                    title_future = executor.submit(title_task)

                # 2. 사용자 메시지 ID 전송 (삭제 버튼용)
                yield json.dumps(
                    {"type": "user_message_id", "chat_id": user_chat.chat_id}
                ) + "\n"

                # 3. LangGraph 스트리밍 시작 (답변 생성)
                for msg, metadata in agent_executor.stream(
                    {"messages": langchain_messages},
                    config=config,
                    stream_mode="messages",
                ):
                    # 틈틈이 제목 생성 다 됐는지 확인 (답변 생성 중에 제목이 완성되면 바로 전송)
                    if title_future and not title_sent and title_future.done():
                        new_title = title_future.result()
                        yield json.dumps(
                            {
                                "type": "history_title",
                                "history_id": history.history_id,
                                "title": new_title,
                            }
                        ) + "\n"
                        title_sent = True

                    curr_node = metadata.get("langgraph_node", "")

                    # (A) AI 텍스트 응답
                    if curr_node == "agent" and msg.content:
                        if not msg.tool_calls:
                            full_ai_response += msg.content
                            yield json.dumps(
                                {"type": "token", "content": msg.content}
                            ) + "\n"

                            # =================================================
                            # [추가] 1.5초마다 중간 저장 (Checkpoint)
                            # =================================================
                            current_time = time.time()
                            # 마지막 저장 후 1.5초가 지났다면?
                            if (current_time - last_save_time) > 1.5:
                                try:
                                    if ai_message_obj is None:
                                        # 아직 DB에 줄이 안 그어졌다면 -> 새로 생성 (Create)
                                        ai_message_obj = Chat.objects.create(
                                            history=history,
                                            type="AI",
                                            content=full_ai_response,
                                            order_num=current_save_order,
                                        )
                                    else:
                                        # 이미 DB에 줄이 있다면 -> 내용만 업데이트 (Update)
                                        ai_message_obj.content = full_ai_response
                                        ai_message_obj.save(update_fields=['content'])
                                    
                                    # 저장 시계 리셋
                                    last_save_time = current_time
                                except Exception:
                                    pass # 중간 저장 실패는 쿨하게 무시 (다음 턴에 하면 됨)

                    # (B) 도구 호출 알림
                    if curr_node == "agent" and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            t_id = tool_call.get("id")
                            t_name = tool_call.get("name")
                            if t_id not in seen_tool_ids:
                                seen_tool_ids.add(t_id)
                                yield json.dumps(
                                    {"type": "tool_call", "tool_name": t_name}
                                ) + "\n"

                    # (C) 도구 실행 결과 저장
                    if curr_node == "tools":
                        content_str = str(msg.content)
                        yield json.dumps(
                            {"type": "tool_result", "length": len(content_str)}
                        ) + "\n"

                        Chat.objects.create(
                            history=history,
                            type="TOOLS",
                            content=content_str,
                            order_num=current_save_order,
                        )
                        current_save_order += 1

                # 4. 스트리밍이 끝났는데 아직 제목이 안 갔다면? (답변이 너무 짧아서 제목보다 빨리 끝난 경우)
                #    여기서 잠깐 기다렸다가 보내줍니다.
                if title_future and not title_sent:
                    new_title = title_future.result() # 끝날 때까지 대기
                    yield json.dumps(
                        {
                            "type": "history_title",
                            "history_id": history.history_id,
                            "title": new_title,
                        }
                    ) + "\n"

                # 5. [AI 최종 답변 저장] - (수정됨)
                if full_ai_response:
                    if ai_message_obj is None:
                        # 한 번도 저장 안 된 짧은 답변일 경우 생성
                        Chat.objects.create(
                            history=history,
                            type="AI",
                            content=full_ai_response,
                            order_num=current_save_order,
                        )
                    else:
                        # 중간 저장이 된 경우 마지막으로 확실하게 업데이트
                        ai_message_obj.content = full_ai_response
                        ai_message_obj.save(update_fields=['content'])

            except Exception as e:
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"
            
            finally:
                # =========================================================
                # [Finally 수정] 중간 저장을 도입했으므로 로직 단순화
                # =========================================================
                try:
                    # 혹시나 에러/중단으로 루프를 빠져나왔을 때, 마지막 잔여물 저장
                    if full_ai_response:
                        if ai_message_obj is None:
                            Chat.objects.create(
                                history=history,
                                type="AI",
                                content=full_ai_response,
                                order_num=current_save_order,
                            )
                        else:
                            # 기존 내용 업데이트
                            ai_message_obj.content = full_ai_response
                            ai_message_obj.save(update_fields=['content'])
                except Exception:
                    pass
                
                executor.shutdown(wait=False)

        return StreamingHttpResponse(
            event_stream(), content_type="application/x-ndjson"
        )

    return JsonResponse({"error": "Method not allowed"}, status=405)


# =========================================================
# API: 삭제 기능 (비회원 지원)
# =========================================================
@csrf_exempt
def delete_message_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        message_id = data.get("message_id")
        if not message_id:
            return JsonResponse({"status": "failed", "message": "message_id is required"}, status=400)

        # ✅ 비회원/세션 기반 히스토리 대응
        if not request.session.session_key:
            request.session.save()

        # 1. 삭제 대상 채팅(Human Message) 찾기
        if request.user.is_authenticated:
            try:
                target_chat = Chat.objects.select_related("history").get(
                    chat_id=message_id,
                    history__user=request.user,
                )
            except Chat.DoesNotExist:
                target_chat = Chat.objects.select_related("history").get(
                    chat_id=message_id,
                    history__user__isnull=True,
                    history__session_id=request.session.session_key,
                )
        else:
            target_chat = Chat.objects.select_related("history").get(
                chat_id=message_id,
                history__user__isnull=True,
                history__session_id=request.session.session_key,
            )

        # HUMAN 메시지가 아니면 삭제 거부 (안전장치)
        if target_chat.type != "HUMAN":
            return JsonResponse(
                {"status": "failed", "message": "Can only delete HUMAN messages"},
                status=400,
            )

        history = target_chat.history
        start_order = target_chat.order_num

        # =================================================================
        # [핵심 수정] 무조건 뒤를 다 지우는 게 아니라, "다음 질문" 앞까지만 지운다.
        # =================================================================
        
        # 1. 내 질문(start_order)보다 뒤에 있는 "다음 HUMAN 질문"을 찾는다.
        next_human_msg = Chat.objects.filter(
            history=history,
            type="HUMAN",
            order_num__gt=start_order  # 현재 번호보다 큰 것 중
        ).order_by("order_num").first() # 가장 가까운 것

        if next_human_msg:
            # 2-A. 뒤에 다른 질문이 있다면? -> 그 질문 직전까지만(range) 삭제
            end_order = next_human_msg.order_num
            Chat.objects.filter(
                history=history,
                order_num__gte=start_order, # 나 포함
                order_num__lt=end_order     # 다음 질문 미만 (<)
            ).delete()
        else:
            # 2-B. 뒤에 질문이 없다면? (마지막 대화인 경우) -> 기존처럼 뒤에 싹 다 삭제
            Chat.objects.filter(
                history=history,
                order_num__gte=start_order
            ).delete()

        return JsonResponse({"status": "success"})

    except Chat.DoesNotExist:
        return JsonResponse(
            {"status": "failed", "message": "Message not found or unauthorized"},
            status=404,
        )
    except Exception as e:
        return JsonResponse({"status": "failed", "message": str(e)}, status=500)


# =========================================================
# API: 새 대화 (비회원 지원)
# =========================================================
def new_chat(request):
    """
    새 대화방 생성 함수 (시간 기준 판단 + 재활용 시 최상단 이동)
    1. '가장 최근에 생성된(created_at)' 방을 찾습니다.
    2. 그 방이 비어있으면 -> 그 방의 순서(order)를 1등으로 높이고 재활용합니다.
    3. 그 방에 대화가 있으면 -> 진짜 새 방을 만들고 순서를 1등으로 줍니다.
    """

    user = request.user
    target_history_qs = None  # 쿼리셋을 담을 변수

    # 1. 대상 쿼리셋 설정 (회원/비회원 분기)
    if user.is_authenticated:
        target_history_qs = ChatHistory.objects.filter(user=user)
    else:
        if not request.session.session_key:
            request.session.save()
        session_id = request.session.session_key
        target_history_qs = ChatHistory.objects.filter(session_id=session_id)

    # 2. [판단 기준] 가장 최근에 '생성된' 방 찾기 (order 기준 아님!)
    last_created_hist = target_history_qs.order_by("-created_at").first()

    # 3. [순서 결정] 현재 존재하는 방들 중 가장 높은 order 번호 찾기
    # (새로 만들거나, 기존 방을 위로 올릴 때 이 번호보다 커야 함)
    current_max_order = (
        target_history_qs.aggregate(Max("order_num"))["order_num__max"] or 0
    )
    new_top_order = current_max_order + 1

    # 4. 로직 수행
    if last_created_hist and not last_created_hist.chats.exists():
        # A. 최근 방이 있는데, 텅 비어있다 -> "재활용 + 맨 위로 이동"

        # 이미 맨 위라면(순서가 max라면) 굳이 업데이트 안 해도 됨
        if last_created_hist.order_num < current_max_order:
            last_created_hist.order_num = new_top_order
            last_created_hist.save()

        # 해당 방으로 이동
        return redirect("chat:chat_interface")

    else:
        # B. 최근 방에 대화가 있거나, 방이 아예 없다 -> "새 방 생성"
        if user.is_authenticated:
            ChatHistory.objects.create(
                user=user,
                order_num=new_top_order,
                description=f"새 대화 {new_top_order}",
            )
        else:
            # 비회원
            ChatHistory.objects.create(
                session_id=request.session.session_key,
                user=None,
                order_num=new_top_order,
                description=f"게스트 대화 {new_top_order}",
            )

    return redirect("chat:chat_interface")


# =========================================================
# API: 채팅방 순서 변경 (Drag & Drop 결과 저장)
# =========================================================
@csrf_exempt
def update_history_order(request):
    """
    [최적화됨] 프론트엔드에서 [id_A, id_B, id_C] 순서로 ID 리스트를 보내면,
    bulk_update를 사용하여 단 한 번의 쿼리로 순서를 업데이트합니다.
    """
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            ordered_ids = data.get("ordered_ids", [])
            
            if not ordered_ids:
                 return JsonResponse({"status": "success"})

            total_count = len(ordered_ids)

            # 1. 권한 확인 (내 채팅방만 건드려야 하니까 필터 생성)
            if request.user.is_authenticated:
                base_filter = {"user": request.user}
            else:
                if not request.session.session_key:
                    request.session.save()
                base_filter = {
                    "session_id": request.session.session_key,
                    "user__isnull": True,
                }

            # 2. [DB 최적화 핵심] 대상 객체들을 한 번에 메모리로 가져오기 (SELECT WHERE IN)
            #    N번 쿼리 날리는 대신 1번만 날립니다.
            histories = list(ChatHistory.objects.filter(
                history_id__in=ordered_ids,
                **base_filter
            ))

            # 3. 빠른 매칭을 위해 딕셔너리로 변환 {id: 객체}
            history_map = {h.history_id: h for h in histories}
            
            update_list = []

            # 4. 프론트에서 보낸 순서대로 메모리 상의 객체 값 수정
            for index, hist_id in enumerate(ordered_ids):
                try:
                    # JSON 데이터는 문자열일 수 있으므로 int로 변환
                    target_id = int(hist_id)
                except ValueError:
                    continue

                if target_id in history_map:
                    history = history_map[target_id]
                    new_order = total_count - index # 위쪽일수록 높은 번호
                    
                    # 값이 변경된 경우에만 업데이트 리스트에 추가
                    if history.order_num != new_order:
                        history.order_num = new_order
                        update_list.append(history)

            # 5. [DB 최적화 핵심] 변경된 객체들을 한 번에 DB에 저장 (BULK UPDATE)
            #    100개를 바꿔도 쿼리는 딱 1번만 나갑니다!
            if update_list:
                ChatHistory.objects.bulk_update(update_list, ["order_num"])

            return JsonResponse({"status": "success"})

        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)})

    return JsonResponse({"error": "Method not allowed"}, status=405)



# =========================================================
# API: 채팅방 삭제 (목록에서 삭제)
# =========================================================
@csrf_exempt
def delete_history_api(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            history_id = data.get("history_id")

            if not history_id:
                return JsonResponse(
                    {"status": "error", "message": "history_id가 없습니다."}
                )

            # 1) 회원인 경우: user 기준으로만 삭제
            if request.user.is_authenticated:
                deleted_count, _ = ChatHistory.objects.filter(
                    history_id=history_id,
                    user=request.user,
                ).delete()

            # 2) 비회원(게스트)인 경우: session_id + user is null 기준으로 삭제
            else:
                # 세션 키가 없으면 새로 생성
                if not request.session.session_key:
                    request.session.save()
                session_id = request.session.session_key

                deleted_count, _ = ChatHistory.objects.filter(
                    history_id=history_id,
                    session_id=session_id,
                    user__isnull=True,
                ).delete()

            # 실제로 삭제된 게 없으면 에러 응답
            if deleted_count == 0:
                return JsonResponse(
                    {
                        "status": "error",
                        "message": "삭제할 수 있는 대화가 없습니다.",
                    }
                )

            return JsonResponse({"status": "success"})

        except Exception as e:
            return JsonResponse({"status": "error", "message": str(e)})

    return JsonResponse({"error": "Method not allowed"}, status=405)



# =========================================================
# API: 채팅방 제목 수정 (회원 + 비회원 공통)
# =========================================================
@csrf_exempt
def rename_history_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        history_id = data.get("history_id")
        new_title = (data.get("title") or "").strip()

        if not history_id or not new_title:
            return JsonResponse(
                {"status": "error", "message": "history_id 또는 제목이 없습니다."}
            )

        # 회원 / 비회원 기준 동일하게 맞추기
        if request.user.is_authenticated:
            base_filter = {"user": request.user}
        else:
            # 세션 키 없으면 생성
            if not request.session.session_key:
                request.session.save()
            base_filter = {
                "session_id": request.session.session_key,
                "user__isnull": True,
            }

        # description 필드를 채팅 제목으로 사용 중
        updated = ChatHistory.objects.filter(
            history_id=history_id,
            **base_filter,
        ).update(description=new_title)

        if updated == 0:
            return JsonResponse(
                {"status": "error", "message": "수정할 수 있는 대화가 없습니다."}
            )

        return JsonResponse({"status": "success", "title": new_title})

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)})



# =========================================================
# 즐겨찾기 상태
# =========================================================
@csrf_exempt
def toggle_pin_api(request):
    """
    특정 채팅방의 즐겨찾기(is_pinned) 상태를 토글(ON/OFF)합니다.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    try:
        data = json.loads(request.body)
        history_id = data.get("history_id")

        if not history_id:
            return JsonResponse({"status": "error", "message": "history_id required"})

        # 회원/비회원 구분 필터
        if request.user.is_authenticated:
            history = ChatHistory.objects.filter(history_id=history_id, user=request.user).first()
        else:
            if not request.session.session_key:
                request.session.save()
            history = ChatHistory.objects.filter(
                history_id=history_id, 
                session_id=request.session.session_key, 
                user__isnull=True
            ).first()

        if not history:
            return JsonResponse({"status": "error", "message": "History not found"})

        # [핵심] 상태 뒤집기 (True <-> False)
        history.is_pinned = not history.is_pinned
        history.save()

        return JsonResponse({
            "status": "success", 
            "is_pinned": history.is_pinned
        })

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)})