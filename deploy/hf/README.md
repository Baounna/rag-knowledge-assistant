---
title: Internal Knowledge Assistant
emoji: 📚
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Internal Knowledge Assistant

A retrieval-augmented assistant over ~1,994 chunks of GitLab's public employee
handbook. Every claim carries a citation that is **verified in code** against
what retrieval actually returned; a citation the model invents is rendered as
plain text rather than a link, and counted as a failure by the eval harness.

Ask about expenses, travel, leave, hiring, or benefits. Ask something the
handbook does not cover and it will refuse rather than guess.

**Note:** this instance runs without a generation model configured, so answers
are refused while retrieval works fully — expand *"N passages retrieved"* under
any answer to see what hybrid search found and which retriever found it.

Source and evaluation report:
<https://github.com/Baounna/rag-knowledge-assistant>
