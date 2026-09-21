"""CLI notification ownership, structured queueing and last-moment consumption."""


def quiet_results_with_manager_completions(cli, result):
    """Keep canonical manager one-shots alive for their existing completion queue."""
    import queue
    import time

    from tools.bot_mode_dm import message_agent_authorized
    from tools.interrupt import is_interrupted
    from tools.process_registry import process_registry
    from tools.process_registry_notifications import ProcessNotificationBatch

    yield result
    if (getattr(cli.agent, "_bot_mode_manager", False) is not True
            or not message_agent_authorized(cli.agent)):
        return
    budget = process_registry._oneshot_completion_wait_seconds()
    if budget <= 0:
        return
    # Share the existing exit-linger budget; finalization must not wait it twice.
    cli._oneshot_completion_deadline = time.monotonic() + budget
    while not (isinstance(result, dict) and result.get("failed")):
        if is_interrupted():
            raise KeyboardInterrupt
        if isinstance(result, dict):
            cli.conversation_history = result.get("messages", cli.conversation_history)
        # Snapshot BEFORE draining. A finished process can still be publishing its
        # notification; its completion event is set only after queue publication.
        pending = []
        for item in process_registry.list_sessions():
            process = process_registry.get(item["session_id"])
            if (process is not None and process.notify_on_complete
                    and not process._completion_event.is_set()
                    and cli._owns_process_notification({"type": "completion", "session_id": process.id,
                                                       "session_key": process.session_key})):
                pending.append(process)
        cli._drain_process_notifications("cli-oneshot-manager")
        try:
            notification = cli._pending_input.get_nowait()
        except queue.Empty:
            notification = None
        if notification is not None:
            if not isinstance(notification, ProcessNotificationBatch):
                cli._pending_input.put(notification)
                return
            message = notification.render(process_registry)
            if message:
                result = cli.agent.run_conversation(
                    user_message=message, conversation_history=cli.conversation_history,
                )
                yield result
            continue
        if not pending:
            return
        remaining = cli._oneshot_completion_deadline - time.monotonic()
        if remaining <= 0:
            yield {"failed": True, "error": "Manager completion wait expired before worker review",
                   "messages": cli.conversation_history, "final_response": ""}
            return
        # A short event wait lets a fast sibling's notification wake the next turn
        # without waiting for the first/slowest process to finish.
        pending[0]._completion_event.wait(min(remaining, 0.1))


class CLIProcessNotificationsMixin:
    def _owns_process_notification(self, event: dict) -> bool:
        """Whether this session owns a delegation event (pre-compression keys resolve to their continuation; fail closed)."""
        event_key = str(event.get("session_key") or "")
        if event.get("type", "completion") == "completion" and event.get("session_id"):
            from tools.process_registry import process_registry

            process = process_registry.get(event["session_id"])
            if process is not None and process.parent_session_id:
                # CLI tools use a turn UUID for session_key. The spawning stored
                # conversation is the durable owner, including after compression.
                event_key = process.parent_session_id
        current_key = str(getattr(self, "session_id", "") or "")
        if not event_key or not current_key:
            return False
        if event_key == current_key:
            return True
        try:
            session_db = getattr(self, "_session_db", None)
            resolved_key = (
                session_db.resolve_resume_session_id(event_key) if session_db is not None else event_key
            ) or event_key
        except Exception:
            resolved_key = event_key
        return str(resolved_key) == current_key

    def _drain_process_notifications(self, consumer: str) -> None:
        from tools.process_registry import process_registry
        from tools.async_delegation import claim_event_delivery, complete_event_delivery
        from tools.process_registry_notifications import (
            ProcessNotificationBatch, TimelineNotification, group_process_notifications)

        claimed = []
        for event, text in process_registry.drain_notifications(
            session_key=getattr(self, "session_id", "") or "", owns_event=self._owns_process_notification,
        ):
            claim = claim_event_delivery(event, consumer)
            if claim is None:
                continue
            claimed.append((event, text))
            complete_event_delivery(event, claim)
        for notifications in group_process_notifications(claimed):
            event, text = notifications[0]
            if event.get("type", "completion") == "completion":
                pending = ProcessNotificationBatch(notifications)
            else:
                pending = TimelineNotification.for_delegation(text, event) if event.get("type") == "async_delegation" else text
                from agent.notification_presentation import diagnostic_process_event
                if diagnostic_process_event(event) and not isinstance(pending, TimelineNotification):
                    pending = TimelineNotification(text, text, "internal_notification", "diagnostic")
            self._pending_input.put(pending)

    def _tui_unwrap_input(self, user_input):
        """Unwrap ``_VoiceInputMessage`` / ``_SeededQueryMessage`` -> ``(text_or_tuple, is_voice_input, is_seeded_query)``."""
        from cli import _VoiceInputMessage, _SeededQueryMessage
        from tools.process_registry import process_registry
        from tools.process_registry_notifications import (
            PROCESS_COMPLETE_DISPLAY_KIND, ProcessNotificationBatch, TimelineNotification)
        if isinstance(user_input, ProcessNotificationBatch):
            rendered = user_input.render(process_registry)
            user_input = rendered and TimelineNotification(
                rendered, user_input.display_text(process_registry), PROCESS_COMPLETE_DISPLAY_KIND)
        # Voice-transcribed messages arrive wrapped in a sentinel so only genuine STT output gets the voice
        # prefix (#65827).
        is_voice_input = isinstance(user_input, _VoiceInputMessage)
        if is_voice_input:
            user_input = user_input.text
        is_seeded_query = isinstance(user_input, _SeededQueryMessage)
        if is_seeded_query:
            user_input = (user_input.text, user_input.images) if user_input.images else user_input.text
        return user_input, is_voice_input, is_seeded_query
