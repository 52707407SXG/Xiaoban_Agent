---
name: live-sports-scores
description: Get live or just-finished sports scores (football/World Cup) reliably. Avoids JS-heavy sports sites by hitting official APIs directly.
category: research
metadata:
  xiaoban:
    requires_toolsets: [web, skills]
    auto_context:
      priority: 100
      max_chars: 7000
      patterns:
        - "(世界杯|World Cup|FIFA|足球|soccer|football).*(比分|score|结果|result|实时|live|完场|final|FT|赛程|fixture)"
        - "(比分|score|结果|result|实时|live|完场|final|FT|赛程|fixture).*(世界杯|World Cup|FIFA|足球|soccer|football)"
        - "(加拿大|Canada|瑞士|Switzerland).*(比分|score|结果|result|实时|live|完场|final|FT)"
---

# Live Sports Scores

When a user asks for a live or recent football (soccer) score, especially World Cup matches, do not start with generic extraction of sports sites. That path wastes turns and can return misleading cached data. Go to official or primary data first.

Mainstream sports sites such as ESPN, FOX, Flashscore, LiveScore, and SofaScore are often JS-rendered SPAs. `web_extract` can return empty page chrome or stale partial scores that look final. `web_search` snippets can also lag for just-finished matches. Both can mislead if used without a primary-source check.

## Football World Cup Matches

Use the FIFA live API directly:

```text
https://api.fifa.com/api/v3/live/football/{competition_id}/{season_id}/{stage_id}/{match_id}?language=en
```

If the current platform allows terminal execution, `curl` the URL with a normal browser User-Agent. If terminal execution is disabled, use `web_extract` or an equivalent web read tool on the same FIFA API URL. Do not fall back to JS-heavy sports pages just because shell/curl is unavailable.

## Finding The Match ID

The match ID is the last segment of the FIFA match center URL:

```text
https://www.fifa.com/en/match-centre/match/17/285023/289273/400021451
```

`match_id = 400021451`

## Parsing The Response

Key fields:

- `HomeTeam.Score` - home score
- `AwayTeam.Score` - away score
- `HomeTeam.TeamName[0].Description` - home team name
- `AwayTeam.TeamName[0].Description` - away team name
- `MatchStatus` - scheduled/live/finished status
- `MatchTime` - current minute when live
- `HomeTeam.Goals[]` / `AwayTeam.Goals[]` - goal events
- `Winner` - winning team when available

## Pitfalls

- Do not trust stale schedule pages, standings pages, or pre-match previews as live/final scores.
- Do not treat social media descriptions as scores; they often contain predictions or promo copy.
- Wikipedia can lag for just-finished matches.
- The FIFA match page render can be cached. Trust the API response over decorative page content.
- `web_extract` on JS sports sites can return a mid-match partial score that looks final. A known failure mode is extracting a `1-0` snapshot while the actual full-time result later differs.
