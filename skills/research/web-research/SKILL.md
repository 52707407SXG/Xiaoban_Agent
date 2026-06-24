---
name: web-research
description: Structured web research workflow for current factual questions, source comparison, and evidence-backed answers.
category: research
metadata:
  xiaoban:
    requires_toolsets: [web, skills]
    auto_context:
      priority: 10
      max_chars: 7200
      patterns:
        - "(查一下|搜一下|搜索|联网|帮我查|帮我搜|最新|当前|实时|今天|新闻|资料|来源|证据|引用|靠谱吗|真假|对比|怎么做|价格|政策|版本|比分|赛程|结果)"
        - "(latest|current|real[- ]?time|today|news|source|citation|evidence|score|fixture|result|price|policy|version|compare|web research)"
---

# Web Research

This skill is adapted from the open-source `pydantic-ai-skills` web-research skill and SearchClaw's public harness idea: plan first, search deliberately, keep evidence, check conflicts, then synthesize. It is not a separate external service and does not require terminal access.

## When To Use

Use this skill for current factual questions, comparisons, source-backed answers, live/recent facts, public web verification, and any user concern about trust, truthfulness, citations, or whether an answer is reliable.

Do not use it for simple chat, purely local My Stand questions with already provided page/file context, or tasks that need no external facts.

## Workflow

1. Classify the information need:
   - quick lookup: one exact fact, deadline, score, price, policy, version, schedule
   - comparison: two or more products, tools, people, companies, claims, or routes
   - investigation: user is asking whether something is true, reliable, current, or suspicious

2. Plan before searching:
   - write an internal plan with 1-3 subtopics for normal questions
   - use up to 5 subtopics only for broad research
   - choose source priority before the first query

3. Search with specific queries:
   - use Chinese queries first for Chinese users and China-localized facts
   - add English or official-source queries when official data is likely outside China
   - avoid broad noisy searches after the first result set; refine by entity, date, status, or official domain

4. Keep an internal evidence ledger:
   - claim or candidate answer
   - source title/domain/URL
   - source type: official, primary data/API, reputable news, aggregator, blog/social, search snippet
   - timestamp/date/status when available
   - exact number/date/status extracted from the source
   - confidence and conflict notes

5. Resolve conflicts before answering:
   - primary/official/API/raw rows beat summaries, SEO pages, reposts, previews, old pages, and JS page chrome
   - two independent current sources can support a non-official claim
   - if a page extraction is empty or stale but the search snippet carries a fresh status, treat the snippet as evidence but label it as a snippet and cross-check once
   - if conflict remains, say sources conflict and do not pretend certainty

6. Synthesize:
   - answer the user's exact question first
   - include source names or links when trust matters
   - state uncertainty plainly when evidence is incomplete
   - do not dump the tool trail; show only the useful evidence

## Hard Rules

- Do not present a stale snippet, cached page, standings table, old preview, or single unverified summary as confirmed current truth.
- Do not keep searching randomly after contradictory evidence appears; switch to official/primary sources or say the conflict.
- Do not hide a useful answer behind vague uncertainty when evidence is strong enough.
- Do not fake citations, dates, quotes, scores, policy text, prices, or source names.
- For live sports, load and follow `live-sports-scores` first when it matches.
