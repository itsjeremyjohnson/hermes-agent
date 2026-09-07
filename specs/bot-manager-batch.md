# Bot Chat manager delegation

Implement an opt-in backend manager mode for existing canonical Bot Chats. The user authorized implementation and code review first, with live testing afterward. Do not deploy this branch or modify live manager profiles during implementation/review.

## Requirements

- Add `agent.bot_mode_manager`, default false, through the existing config loader and agent initialization. Ordinary bots retain their single-message schema and existing messaging guidance.
- Managers receive guidance to answer small questions directly, delegate substantial work to existing specialists, and dispatch ready independent assignments before ending the turn. Serialize actual dependencies and conflicting writes. Respect the user's scope and review automatic returns.
- Extend the existing injected `message_agent` tool with an `assignments` batch containing target/message pairs. Keep single target/message calls backward compatible. Batches are manager-only and canonical-Bot-Chat-only.
- Bound each batch to eight assignments. Reject malformed batches and mixed single/batch input before dispatching anything. Validate routes with the existing local, peer, and Desktop-relay paths. Return separate indexed results for routing/start failures without blocking other valid entries. Never retry the entire batch or duplicate successful sends.
- Reuse background delivery, automatic completion notifications, sender identity, per-recipient turn locks, and canonical conversation identity. Do not add another worker scheduler. Different recipients must be able to overlap; one recipient retains its existing serialization.
- Keep the setting fixed on an existing agent object. Use the established Bot Chat capability epoch when a new agent loads changed config; do not rewrite history or add per-turn prompt changes.
- No desktop/group routing changes, new agent profiles, infrastructure changes, or live tasks in this implementation.

## Verification

Exercise config loading, generated protocol/schema, containment, malformed input, and partial results through the real inline executor. Use an event/barrier check with real background delivery runners and deterministic fake worker executables: both recipients must start before either is allowed to finish. Existing messaging, relay, and session tests must continue to pass. A later live test must separately verify model task decomposition, worker overlap, automatic returns, and manager review.
