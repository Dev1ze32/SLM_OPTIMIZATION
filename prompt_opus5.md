Use whenever the user is writing, reviewing, or tuning a prompt, system prompt, or agent configuration meant for a high-capability "Opus"-class Claude model, especially Claude Opus 5 or later. Trigger on requests like "help me prompt Opus 5", "write a system prompt for my agent", "why is Opus giving me such long or rambling output", "how do I stop it from going off and doing extra stuff", "tune my Claude Code or subagent setup", or "optimize this prompt for Opus 5". Also trigger when the user pastes an existing prompt written for an older model (e.g. Opus 4.8) and asks whether it needs updating. Covers giving goal-not-steps instructions, full-spec-upfront framing, controlling response length and file-output length, controlling progress narration, scope-boundary enforcement, subagent delegation limits, defining stop and ask points, self-correction narration control, and code-review prompting (report-everything vs report-only-severe).

Prompting Opus-class Models (Opus 5+)
Source and a caveat to pass on

This skill distills tactics from a user-supplied article about prompting "Claude Opus 5." Before treating any of this as ground truth, flag to the user that these specific behavioral claims (verbosity defaults, subagent tendencies, self-correction habits, etc.) come from a secondary source, not verified Anthropic documentation — Claude's own knowledge of current models may be incomplete or out of date. Say this once, briefly, don't belabor it on every use.

That said, the tactics themselves are generically sound prompt-engineering practice for any highly capable, agentic model, so they're worth applying even if the specific "Opus 5" framing turns out to be off.

When to reach for which block

Don't dump every block into every prompt. Pick based on what the user is actually building:

Situation	Blocks to use
Writing a first-message task request	Goal-not-steps + Full-spec-upfront
Output feels too long / rambly in chat	Shorten response length
Files/reports it writes keep bloating	Shorten files written to disk
Too much "I'm now going to..." commentary	Turn down progress narration
It keeps doing more than asked	Stop it from going beyond the task
Using subagents / Claude Code / Cowork	Cap subagents + Define stop points
It corrects itself out loud constantly	Turn down self-correction narration
Prompting for code review	Code review block (see trap below)
Prompting for image/spreadsheet/slide work	Images, spreadsheets and slides notes

Ask the user which of these problems they're actually hitting if it's not obvious — don't assume they want the full kitchen-sink prompt.

The blocks
1. Goal, not a step-by-script

State what you're building, who it's for, and why — let the model derive the steps. Don't hand it a checklist unless the steps genuinely must happen in a fixed order.

Task: what I'm working on overall and who it's for
Why: what this output needs to enable
Need: the specific request in one clear sentence
2. Full spec upfront (for complex work)

Adding constraints mid-stream is expensive — the model has already built a solution around what it heard first.

Task: what I'm working on overall and who it's for
Why: what the output needs to enable
Need: the specific request
Format: how the result should be structured
Boundaries: what must not be touched
Done when: what signals the work is finished

The "Done when" line matters most — undefined completion criteria means the model invents its own.

3. Shorten response length

Effort/thinking settings control depth of reasoning, not output length — don't conflate them.

Keep responses focused and brief with no warm-up.
Keep caveats and disclaimers short and spend most
of the response on the actual answer. When asked
to explain something, give a condensed summary.
A full explanation only if I ask for one.

For long system prompts, also repeat a short version near the end — buried mid-document instructions get less weight:

<tone_preference>
Keep outputs reasonably concise.
</tone_preference>
4. Shorten files written to disk

Separate problem from chat verbosity — reports/docs saved to files tend to bloat with padding.

Keep files at the size the task requires. Don't cut
substance, but don't pad the length with sections
added for volume, repeated summaries of the same
material, or boilerplate openings.
5. Turn down progress narration

Describe the cadence you want (positive framing), not a list of prohibitions.

Before your first action, say in one sentence what
you're about to do. After that, report only when you
find something important or change approach. At the
end, lead with the result: the first sentence answers
what was done and what was found, details come after.
6. Stop it from going beyond the task

The single highest-value block for agentic/coding work — prevents scope creep.

Do what was asked, within the boundaries that were
asked for. Make routine calls yourself. Only ask when
different readings of the request would materially
change the result. If the request looks mistaken or
a better path exists, say so in one sentence and do
it as asked. Don't quietly narrow, widen or replace
the task. Finish the scope you were given and don't
go past it.
7. Cap subagents

Only relevant in Claude Code / Cowork / anything with subagent access.

Only hand subagents large work that is genuinely
independent and splits into parallel tracks, such as
sweeping many files at once. Don't delegate what you
can finish yourself in a few steps. Don't spin up
subagents to check your own work. If one is enough,
don't launch several.

Simpler alternative: just state a hard cap ("use at most 2 subagents") in the first message.

8. Define stop points for autonomous runs
Stop and ask me only in three cases: an irreversible
or destructive action, a real change in the scope of
the task, or something only I can provide. Otherwise
keep working and report when done.
9. Turn down self-correction narration

Narrow the condition rather than banning it outright — you still want it to flag real errors.

Only revisit an earlier statement if the error changes
code, conclusions or a decision. State the correction
briefly and keep working. Minor slips that change
nothing, fix silently.
10. Code review — avoid the under-reporting trap

If a prompt says "report only high-severity issues" or "be conservative," the model takes it literally and under-reports. Fix: let it surface everything, filter afterward.

List everything that looks questionable and rank it
by importance. I'll decide what gets fixed.
11. Delete these from old prompts

These were compensations for older-model unreliability and may now just add noise/cost:

"verify your answer before responding"
"add a final verification step"
"use a subagent to double-check"
"re-read your output for errors" Replace with nothing — don't substitute a different verification instruction unless the user has a specific reason to keep one.
12. Images, spreadsheets, slides
Vision-compensation instructions built into older prompts may no longer be necessary — worth re-testing without them.
For visual self-checking, prefer handing over a file path so the model can open/crop/zoom itself, rather than pasting a static image and asking for a verbal impression.
For spreadsheets/slides: give it a finished example to match rather than a prose description of style ("match this file's formatting" beats "use blue headers and Calibri").
Composing a full system-prompt addendum

When the user wants "the whole package" (e.g. for a Claude Code config or agent system prompt), combine relevant blocks under labeled headers rather than running them together as one paragraph — this keeps each instruction independently scannable and easy for the user to delete pieces of later. Don't include every block by default; ask which failure modes they're actually seeing, or infer from context (e.g. they're building a coding agent → include scope boundaries, subagent cap, stop points; they're doing writing tasks → include length control blocks only).

A note on the "delete old prompts" instinct

Resist over-applying this. If the user's older instructions exist for a different reason than compensating for weaker models (e.g. a genuine compliance requirement to double-check outputs before they ship), don't remove them just because this skill says older-model verification instructions are usually redundant now. Ask if unsure.