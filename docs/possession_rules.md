# Our possession rules

The football rules our algorithm implements. Agreed 6 Nov 2026 (Step 5). Each rule names the
evidence behind it. This becomes the "possession rules" section of the technical document.

Source of the definitions: CHALLENGE.md (the organisers' description). Evidence: our own analysis
of the public 5 minutes (understanding only), the tracking health check, and 7 open Bundesliga
matches (IDSSE/DFL, CC-BY 4.0). No rule or constant is tied to specific frames of the match.

## Rule 1 — Three states, always exactly one
Every graded frame is H (home ball, alive), A (away ball, alive) or D (dead). No "contested"
state: a loose ball belongs to the last team that controlled it.

## Rule 2 — When the ball goes dead
Ball crosses a touchline or goal line, a goal is scored, or the referee stops play (foul,
offside, injury). **For fouls, the ball is dead from the moment of the foul**, not the whistle
(decision D17; to be verified as the convention in the Bundesliga labels).

Detected by combining evidence (D10): ball outside the lines; ball slow or still; all players
slowing and reorganising; a long ball blackout while players are slow; Smart Tagging set piece
about to start.

Stoppages are never brief: under 2 s in 0% of our public dead balls and 1% of Bundesliga ones.
A one-second "dead blip" inside play is noise.

## Rule 3 — Restarts and who gets the ball
Play restarts with a set piece; the restarting team gets possession.

| Restart | Taken by | Recognised by |
|---|---|---|
| Throw-in | team that didn't touch it last | ball at a touchline; the player there |
| Corner | attacking team | ball at a corner flag |
| Goal kick | defending team | ball in the 6-yard box |
| Free kick | team fouled | ball still, opponents forming a wall |
| Kick-off | team that conceded (or the start) | ball on the centre spot, teams in own halves |

Set-piece type comes from Smart Tagging when it fires, and from ball geometry when it doesn't
(tagging is clearly incomplete: only 5 free kicks and 26 throw-ins tagged).

**Throw-ins:** Smart Tagging never names the team, so combine the player at the touchline with
"the team that did not have it before the ball went out" (D19).

Fallback: the restart goes to the other team in 70–75% of stoppages (our 5 min: 75%; Bundesliga 70%).

**Goals:** a long ball blackout ending with the ball on the centre spot is a goal stoppage: the
whole blackout is dead, and the team that conceded (opposite of the team in possession before
the stoppage) kicks off. In our tracking all 4 blackouts ending at the centre spot coincide with
Smart Tagging kick-off tags (3 after goals, lasting 43–221 s; one second-half kick-off); the tags
fire about 3 s before the ball reappears. Implemented as a general rule, never as specific frames.

## Rule 4 — Open-play changes need control
Possession changes only when an opponent controls the ball: he receives it (ball slows beside
him) and then makes a deliberate pass to a teammate or carries it. **However short, a brief spell
counts if it contains such a deliberate action** (D16); a ricochet off a player does not.

Not a change: a touch without control; a deflection that does not stop the pass; a 50/50 or
header that goes nowhere; an aerial duel won without that team then controlling it.
During loose or contested balls (e.g. after a corner), possession stays with the last team in
control.

## Rule 5 — When the ball is missing
41% of graded frames, mostly in blackouts over 10 s.
- Same team before and after, no stoppage signs: that team kept it.
- Players walking and reorganising: probably a stoppage (Rule 2).
- Different team after: place the change using team shape (attackers push up, defenders drop)
  and Smart Tagging phases.
- Unsure frames: two versions were built and compared on the external scoreboards (D18):
  (a) write X when unsure, (b) always guess. **Always guess won (D40):** X inside a possession leaves
  spellF1 unchanged (the scorer fills X with the previous label) and costs macroF1, and X on doubtful
  short pieces gained nothing measurable.

## Rule 6 — Distrust bad tracking
Ignore impossible ball jumps (0.3% of frames). Reason about teams, never individual players
(slots swap identity; duplicates in 13% of frames). Ball speed comes from positions (D11).

## Rule 7 — Stay inside the competition rules
Kick-offs and half starts are detected by the algorithm, never typed in from the video. The same
settings apply to every match. All tuning is done on external data (D12, D14).
