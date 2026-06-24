---
name: chinese-web-research
description: Web research workflow for Chinese-speaking users; prefer Chinese sources that already converted timezone, currency, and local context.
category: research
metadata:
  xiaoban:
    requires_toolsets: [web, skills]
    auto_context:
      priority: 50
      max_chars: 5600
      patterns:
        - "(查一下|搜一下|搜索|联网|最新|当前|实时|今天|新闻|赛程|比分|结果|价格|政策|人民币|北京时间|中文来源|国内)"
        - "(世界杯|足球|比赛|赛程|比分|开赛|完场|直播|积分榜)"
---

# Chinese Web Research

## Mandatory Trigger

Load this skill before searching when a Chinese-speaking user asks about:

- sports live scores, fixtures, schedules, match results, standings, kickoff times, or TV broadcasts
- currency, RMB prices, local availability, Chinese market data
- policies, versions, releases, or public facts that should be answered in Chinese terms

## Core Rule

Search Chinese sources first when Chinese sources are likely to have already converted timezone, currency, or local naming. Do not manually convert from English sources when a reliable Chinese source already published the localized answer.

## Workflow

1. Search one Chinese query first. Add one English/official query only if needed.
2. Prefer Chinese source results for Beijing time, RMB, Chinese names, and local wording.
3. For live scores, avoid burning turns on JS-only Western sports pages. Use server-rendered Chinese sports portals, official match centers, or the `live-sports-scores` workflow.
4. Use search-snippet fallback only when extraction fails and the snippet contains explicit score/date/status markers. Label it as snippet evidence and cross-check once.
5. If manual conversion is unavoidable, cross-check against a Chinese source.

## Pitfalls

- Do not confuse venue local time, ET/EDT, and Beijing time.
- Do not treat old schedule pages, standings, or pre-match previews as current scores.
- Do not keep retrying JS-heavy pages that extraction cannot read.
- If sources conflict, answer with the source conflict instead of inventing a clean result.
