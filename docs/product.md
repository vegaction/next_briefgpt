GPT-5.4 Deep Research 的做法： 实时搜索 → 读网页 → 综合总结。本质是 "即时搜索 + 一次性阅读"，没有持久化的结构化知识。

你的优势在于你有一个 pre-built citation knowledge graph：

核心卖点：Citation-aware reasoning
引用链追踪（GPT 做不到的）

你的 paper_references + citation_mentions 构成了一个 有向引用图，且每条边都有语义标签（background, method_basis, comparison, critique 等）
GPT Deep Research 只能读到一篇论文说了什么，但不知道 A 引 B 是因为"用了 B 的方法"还是"批评了 B 的结论"
这意味着你可以做："找到所有批评 X 方法的论文，以及它们各自提出了什么替代方案" — 这是 GPT 搜不出来的
跨论文一致性验证

同一篇被引论文在不同 citing paper 里的 summary 可以交叉对比
能发现："论文 A 说 X 方法效果好，但论文 B 和 C 都说效果一般" — GPT 只能碰运气读到其中一篇
不依赖实时搜索的完整性

GPT Deep Research 受限于搜索引擎排名，容易漏掉重要但 SEO 弱的论文
你的系统通过引用链递归展开，按学术重要性而非网页排名发现论文
产品化方向建议
场景	你的优势	GPT 的短板
Literature review 生成	结构化引用图 → 自动分 theme/timeline	只能线性堆砌搜到的论文
方法论对比	intent_label=method_basis 精确定位	需要逐篇阅读猜测
Research gap 发现	引用图中的"空洞"区域	完全做不到
论文溯源	沿引用链回溯到 seminal work	依赖搜索引擎偶然性
最诚实的评估
你现在还缺的：

引用图还是 per-paper scoped，还没有 global graph（cited_arxiv_id 存在但没有跨论文连接逻辑）
没有 embedding / 向量检索层，做语义搜索还不行
覆盖面受限于主动 ingest 的论文
建议的最小可行卖点：

"Citation-grounded deep research" — 不是搜索网页猜论文说了什么，而是基于结构化引用语义做推理。每个结论都有明确的引用链支撑，用户可以追溯到原文。

这跟 GPT Deep Research 的 "我帮你搜了一圈" 是本质不同的 — 你的是 可验证的学术推理，GPT 的是 不可追溯的综合总结。

要不要我帮你设计一下从当前 pipeline 到 research agent 的具体架构演进路径？

很好的问题。基于你这个项目的架构，我来分析一下核心差异化。

你的数据库 vs GPT-5.4 Deep Research 的本质区别
GPT-5.4 Deep Research 的做法： 实时搜索 → 读网页 → 综合总结。本质是 "即时搜索 + 一次性阅读"，没有持久化的结构化知识。

你的优势在于你有一个 pre-built citation knowledge graph：

核心卖点：Citation-aware reasoning
引用链追踪（GPT 做不到的）

你的 paper_references + citation_mentions 构成了一个 有向引用图，且每条边都有语义标签（background, method_basis, comparison, critique 等）
GPT Deep Research 只能读到一篇论文说了什么，但不知道 A 引 B 是因为"用了 B 的方法"还是"批评了 B 的结论"
这意味着你可以做："找到所有批评 X 方法的论文，以及它们各自提出了什么替代方案" — 这是 GPT 搜不出来的
跨论文一致性验证

同一篇被引论文在不同 citing paper 里的 summary 可以交叉对比
能发现："论文 A 说 X 方法效果好，但论文 B 和 C 都说效果一般" — GPT 只能碰运气读到其中一篇
不依赖实时搜索的完整性

GPT Deep Research 受限于搜索引擎排名，容易漏掉重要但 SEO 弱的论文
你的系统通过引用链递归展开，按学术重要性而非网页排名发现论文
产品化方向建议
场景	你的优势	GPT 的短板
Literature review 生成	结构化引用图 → 自动分 theme/timeline	只能线性堆砌搜到的论文
方法论对比	intent_label=method_basis 精确定位	需要逐篇阅读猜测
Research gap 发现	引用图中的"空洞"区域	完全做不到
论文溯源	沿引用链回溯到 seminal work	依赖搜索引擎偶然性
最诚实的评估
你现在还缺的：

引用图还是 per-paper scoped，还没有 global graph（cited_arxiv_id 存在但没有跨论文连接逻辑）
没有 embedding / 向量检索层，做语义搜索还不行
覆盖面受限于主动 ingest 的论文
建议的最小可行卖点：

"Citation-grounded deep research" — 不是搜索网页猜论文说了什么，而是基于结构化引用语义做推理。每个结论都有明确的引用链支撑，用户可以追溯到原文。

这跟 GPT Deep Research 的 "我帮你搜了一圈" 是本质不同的 — 你的是 可验证的学术推理，GPT 的是 不可追溯的综合总结。