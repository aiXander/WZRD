---
name: video-prompting
description: "Writing image→video (kling) prompts: start/end keyframe segments, static camera, loop closure, projectability check. Load before any image→video call."
metadata:
  kind: guide
---

# Video prompting for projection

The clip will be background-subtracted against the static surface, so the
only thing that may change between frames is the projected content itself.

## Every call

- **Pass keyframes as arguments**: start frame as `image_url`, end frame as
  `tail_image_url` (Kling v3: `start_image_url` / `end_image_url`). Both,
  whenever the segment goes keyframe → keyframe.
- **Static camera, static background.** Say it plainly: "Locked-off static
  camera. The background stays dark and completely unchanged; only the
  character and its light move." Ask for additive, luminous light on the
  foreground and subject.
- **Describe the motion as if no keyframes existed.** Never "transition from
  frame 1 to frame 2"; describe what happens, concretely and in order: who
  moves where, how, at what pace, what lights up as they pass.
- **Duration** `"5"` unless the user asks otherwise; 10 s for slow, long
  actions.
- Keep `negative_prompt` covering blur and distortion, and add "camera
  movement, zoom, pan, background change, flicker".

## Loops

A seamless loop is one more segment from the last keyframe back to the first,
then `concat_videos(..., seamless_loop=true)`. Describe that last segment as the
natural continuation that returns the scene to its opening state.

## Is the clip projectable?

`view_media` the clip: the background must match across the contact sheet. Then
`subtract_background_video` against the start frame; a clip whose result lights
up the whole frame has camera drift or flicker and must be regenerated, however
good it looks raw.
