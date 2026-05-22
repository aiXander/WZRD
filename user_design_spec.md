# user_design_spec.md

> A subjective statement of what I want from WZRD as a human being who plays live shows and builds physical installations. Not architecture, not tech stack — just goals, feelings, and the shape of the experience I'm trying to build for myself. Use this as the north star when making design trade-offs, especially as we start designing the Tauri shell (Phase 4 and beyond).

---

## What WZRD actually is, in one sentence

A tool for making a *specific physical thing* — a tree, a rock formation, a facade, a stage prop — come alive with projected light, merging dream with reality. The surface is the instrument; light is how I play it; segmentation is the alphabet.

That sentence is the soul of the project. Every UX decision should make that sentence *easier to feel*, not harder.

---

## Who I am in this loop

There are actually two of me, and the tool has to serve both.

**Daylight-me, on a ladder, the day before the show.** I'm photographing the surface, walking around it, identifying what counts as "trunk" vs "branches" vs "leaves," naming regions, aligning the projector, dialing in calibration. I'm patient here — it's slow, deliberate, often outdoors, often alone. The output of this work is a *layer pack* that locks the artistic vocabulary for the venue.

**Showtime-me, at the booth, with the music up.** The surface is alive in front of a crowd. I'm tuning, triggering, summoning new effects with the AI, throwing things away. I am not a programmer in this moment. I am a performer with one hand on a knob and one eye on the wall.

The tool fails if it confuses those two modes — if it makes daylight-me rush, or makes showtime-me think like an engineer.

---

## What I actually want from this tool

### 1. The physical surface is the canvas, and the canvas is the constraint I'm playing with

Other VJ tools treat the projection surface as a passive screen. WZRD has to treat it as a *collaborator*. The cracks, the leaves, the windows — those are not obstacles to map around; they are the shapes I'm animating. The whole point is that "the dark stays dark and merges back into the night." Anything in the UX that nudges me toward thinking in rectangular video frames, or that hides what region I'm currently affecting, betrays the thesis.

When I'm looking at the screen, I should always be able to see *the surface* — the photo, the masks, the named regions — as the primary visual. Effects are described in terms of what they do *to that surface*. The composite preview is secondary; the surface map is primary.

### 2. Let me speak in surface-language, not screen-language

I never want to address a layer by index. I want to say "the trunk," "all the leaves," "the left eye," "everything tagged 'window'," and the tool should know what I mean. Tags, ids, groups, parents — these are the words of the instrument. Whatever the system internally calls things, the surface I touch should always be the *names I gave the regions* during setup.

If I re-segment the surface next year because a branch fell off, my old scenes should still work — because they reference "leaves," not "blob 17."

### 3. Make my imagination the bottleneck, not the effect library

When I picture something — *"the trunk should look like slow rising sap with golden veins, the leaves should shimmer faintly between deep green and forest teal, every fourth bar one random leaf cluster blooms outward in white"* — I want to type that, speak it, or describe it in a few words, and see it on the surface within seconds. I do not want to scroll a menu of pre-baked effects. The AI's job is to *write the shader*, not pick from a list.

If I can describe it clearly, the system should be able to render it. The shipped effects are starting points, never ceilings.

### 4. Tune by feel, not by re-prompting

Once an idea exists on the surface, I want to shape it with my hands — sliders, colour pickers, a knob, a tap. *"Slightly faster. Pull the green colder. Make the bloom last half as long."* These are physical adjustments. I do not want to go back to a chat box and ask the AI to change a number; that's too slow, too disembodied, and breaks the flow of performing. The AI authors the instrument; I play it.

Every parameter the AI exposes should end up as something I can grab live without re-prompting.

### 5. Never break the show while I'm exploring

I need to be able to experiment, fail, retry, throw away — *while the surface keeps performing*. The audience should never see my drafts, never see a black flash because I saved a broken shader, never see a "compile error" dialog. There has to be a clean separation between the place where I'm authoring and the place where I'm performing, and the projector should keep its last good frame whenever something breaks behind the scenes.

Hot-reload that *isn't* "swap on success" would actively harm the show. This is non-negotiable.

### 6. Lock in to the music

The surface has to *listen*. Bass, snare, hats, beats, tempo — whatever the music is doing, the regions should feel coupled to it without me babysitting every parameter. When I write a new effect with the AI, audio reactivity is part of how I describe it from the start. *"Trunk brightness pulsing on the kick. Leaves shimmer rate tied to the hats."* — should be sentences, not configuration screens.

That said, audioreactivity can become too much. A master "how much does the surface listen right now" knob is always the final boss and must never be bypassed. The crowd should feel the music in the light; they should not feel a metronome.

### 7. Pre-show setup must feel like preparation, not configuration

Daylight-me needs a calm, focused workflow for the slow stuff: take the photo, prep the surface, segment, name regions, align, calibrate. This is the slow half of the tool and it deserves its own room — not the same UI as live performance, not buried in menus next to sliders.

The output of this room is a *pack* I can name, save, version, and trust will load identically tomorrow night at the next venue. If I open a pack from last year, I should be able to project it within minutes.

### 8. Calibration recovery in under a minute, always

Projectors get bumped. Wind moves the rig. Someone leans on a stand. When that happens during a show, I need to fix the alignment by dragging four corners on my phone in under a minute, and never re-segment. The tool should make this feel like tuning a guitar string, not re-recording the album.

### 9. Trust it to stay up

This thing is going to run at venues, often outdoors, often on a flaky internet connection, often after I've been on my feet for hours. It needs to come back up cleanly when the power blinks. It needs to stay stable for hours of continuous use. It needs to not require SSH-debugging at 2am. If something fails, the projector should degrade gracefully — last good frame, last good scene, last good pipeline — not go black, not lock up. Reliability is part of the artistic experience.

### 10. Stay legible to me at a glance

When I look at the operator UI mid-set, I need to know: what scene is playing, which regions are active, how loud the room feels to the system, whether audio / MIDI / OSC are connected, whether the GPU is keeping up, whether anything just hot-reloaded. No deep menus, no jargon — the information I need to perform should be on one screen, big enough to read in low light, responsive to a phone or tablet.

### 11. Collaborate with me — don't replace my taste

The AI is a co-author, not a curator. I bring the aesthetic; it brings the speed and the shader fluency. I want it to write what I asked for, not what it thinks I should want. When it gets something wrong, it should be easy and fast to correct ("no, slower, and warmer") without unwinding the whole conversation. When it gets something right, I should be able to keep that effect, name it, and reuse it forever.

The AI's natural failure mode is genericness. Push the tool toward *my* visual vocabulary, not the median of its training data.

### 12. Persist what's good, across venues

Scenes I love are scenes I want back — at this gig, at the next one, six months from now at a different installation entirely. I want to save scenes, save authored effects, save calibration, save the whole show, and trust they'll come back exactly as I left them. The same tree pack should be playable identically next summer. A leaf effect I wrote for a tree should be reusable on a building facade if the binding semantics line up.

The unit of reuse is the *effect file* and the *scene*; both should live as plain files I can copy, share, version-control, and trust.

### 13. Mobile-first for performance, desktop-first for authoring

Setup work needs a real screen, a keyboard, Monaco, room to think. That's a laptop job.

Performance work needs to follow me around the room — to the booth, to the floor, to a ladder to look at the surface, to a chair when I'm tired. The live-performance surface (scene switching, slider rack, audio-debug viz, calibration nudges, panic master-fader) must run cleanly on my phone or tablet. I should not be physically chained to the laptop during a show.

### 14. Auto-pilot mode for when I'm human

I sometimes need to pee, get a drink, talk to people, sleep. The system needs a chain-of-scenes auto mode where I can pre-queue good looks, decide how many bars or how many minutes each one runs, and walk away with confidence that the surface will stay alive and varied without me. Bonus: the AI can riff lightly within the queue's constraints to keep things from looping too obviously.

### 15. Surprise me, sometimes

The best moments in a live set are the ones I didn't plan. If the AI can occasionally suggest a direction, riff on what's already playing, or push an idea further than I asked — without overstepping, and clearly distinguishable from things I asked for — that's a win. The tool should be capable of being a creative provocateur, not just a transcriber of my requests.

### 16. Two-process, two-rooms is fine — but it has to feel like one tool

Architecturally, WZRD is a Python offline pipeline plus a Rust realtime engine plus a Tauri shell plus (eventually) an MCP agent. I do not care, and the user-facing experience must hide this. Opening a pack, generating a new texture with the AI, segmenting a fresh photo, dropping a clip onto a region — these should feel like single gestures, not "switch terminals, wait for the server, copy the URL, paste it into the scene file."

If I'm ever aware that two processes are talking to each other, the UX has failed.

---

## The feel I'm chasing

A live, responsive, intimate instrument for *making physical things glow*. Less "control panel," less "video software," more "synth with a body." Something that disappears into the performance — where I stop thinking about the tool and start thinking about the surface and the room.

Daylight-me should feel like a craftsman at a workbench. Showtime-me should feel like a musician at an instrument. The tool should never make either one feel like an engineer.

If the system ever forces me to leave performance-mind and enter engineer-mind during a show, it has failed. Every design decision should be checked against that line.

---

## What I'm NOT trying to build

- **A general-purpose video-mapping suite.** MadMapper, Resolume, HeavyM exist and are better at being themselves. I am not chasing parity with them. The thing they don't do — *semantic per-region binding on a real photographed surface* — is the only thing I owe the user.
- **A 3D scene editor.** No cameras, no objects, no materials in the Blender sense. The surface is 2D, photographed, segmented. That's the whole model.
- **A pixel-mapping suite for LEDs / strips / matrices.** Different tool, different repo.
- **A tool optimised for studio production / pre-rendered video output.** Recording is explicitly out of scope. WZRD drives a projector in real time. If you want a video file, screen-record the projector.
- **A tool that requires a tech in the loop to operate.** I run this alone, often outdoors, often after a long day. One operator, one laptop, one projector, one phone.
- **A scene format authored only through a GUI.** Plain JSON on disk is canonical. The GUI is one of several authoring surfaces (alongside the AI, alongside a text editor, alongside another agent). The tool must remain usable headlessly so the agent loop never breaks.
- **A library of pre-made looks.** The thesis is that effects are *written*, by me or by the AI, against this specific surface. A built-in catalog beyond a small reference set would actively undermine that thesis.

---

## How to use this document

When making technical or UX trade-offs, ask: *does this make the experience above easier or harder for either daylight-me or showtime-me?*

- If a clean abstraction makes the live tool slower to respond, slower to iterate, or less expressive — pick a different abstraction.
- If a feature is convenient for the engineer building it but forces the performer to think like an engineer — cut it.
- If a setup step feels rushed by the tool — slow the UI down and give it room.
- If an authoring path requires the GUI to exist — find the headless path first and treat the GUI as ergonomic sugar.
- If the dark parts of the surface ever stop being dark — stop and fix that before anything else.

The surface is the instrument. The performer's experience is the thing being optimised.
