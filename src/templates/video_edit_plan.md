# Video Edit Plan

Analyze the following recorded video segments and generate editing decisions for each part.

For each part, decide:
- **speed_factor**: Playback speed multiplier (1.0 = normal, >1.0 = faster, <1.0 = slower)
- **trim_start_s**: Seconds to trim from the beginning
- **trim_end_s**: Seconds to trim from the end
- **remove_dead_time**: Whether to detect and speed up frozen/idle frames

## Guidelines
- Browser recordings often have idle time at start/end - suggest trimming
- Title cards should keep speed_factor=1.0
- AI-generated clips rarely need trimming
- Consider the target duration when setting speed factors
- Keep speed_factor between 0.5 and 2.0 for natural-looking playback
- Only set remove_dead_time=true for browser recordings that may contain idle periods
- Provide brief notes explaining the reasoning for each decision

## Output Format
Return a JSON array of decisions:
```json
[
  {
    "part_id": "part-1",
    "speed_factor": 1.2,
    "trim_start_s": 0.5,
    "trim_end_s": 0.3,
    "remove_dead_time": true,
    "notes": "Browser recording with idle start"
  }
]
```
