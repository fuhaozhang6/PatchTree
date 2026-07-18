% Chinese working draft in the official AAAI-27 layout.
% !TeX program = xelatex
\documentclass[letterpaper]{article}
\usepackage[submission]{aaai2027-zhdraft}

\usepackage[hyphens]{url}
\usepackage{graphicx}
\urlstyle{rm}
\def\UrlFont{\rm}
\usepackage{natbib}
\setcitestyle{numbers}
\usepackage{caption}
\frenchspacing

\usepackage{amsmath,amssymb,mathtools,bm}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{fvextra}
\usepackage{fontspec}

% Chinese support for the working draft only. The final English submission
% should remove fontspec and these font declarations, then use pdfLaTeX.
\setmainfont[
  Path=fonts/,
  BoldFont=NotoSansHans-Bold.otf,
  ItalicFont=NotoSansHans-Regular.otf,
  BoldItalicFont=NotoSansHans-Bold.otf
]{NotoSansHans-Regular.otf}
\setsansfont[
  Path=fonts/,
  BoldFont=NotoSansHans-Bold.otf,
  ItalicFont=NotoSansHans-Regular.otf,
  BoldItalicFont=NotoSansHans-Bold.otf
]{NotoSansHans-Regular.otf}
\setmonofont[
  Path=fonts/,
  BoldFont=NotoSansHans-Bold.otf,
  ItalicFont=NotoSansHans-Regular.otf,
  BoldItalicFont=NotoSansHans-Bold.otf
]{NotoSansHans-Regular.otf}
\XeTeXlinebreaklocale "zh"
\XeTeXlinebreakskip=0pt plus 1pt

\RecustomVerbatimEnvironment{verbatim}{Verbatim}{
  fontsize=\scriptsize,
  breaklines=true,
  breakanywhere=true,
  frame=single,
  rulecolor=\color{black!15},
  framesep=1.5mm
}

\providecommand{\tightlist}{%
  \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\hypertarget}[2]{#2}
\renewcommand{\abstractname}{摘要}

\ifdefined\pdfinfo
\pdfinfo{
/TemplateVersion (2027.1)
}
\fi

\setcounter{secnumdepth}{2}

\title{PatchTree: Type-Guided Patch Aggregation and Merge Tree Pruning for Skill Optimization}
\author{Anonymous Submission}
\affiliations{}

\begin{document}
\maketitle

\begin{abstract}
Agent Skill 以自然语言文档封装任务流程、工具策略和行为约束，为大语言模型提供了一种可解释、可迁移的外部能力适配方式。近期研究开始利用执行轨迹和任务反馈自动构建或持续优化 Skill，但从样本级反馈到全局 Skill 更新仍存在关键的聚合困难：单样本 Patch 往往具有较强的局部性，容易产生碎片化和样本记忆；批量 Patch 归纳则可能形成冲突或过度泛化的规则。本文提出一种类型引导的簇级 Patch 聚合与合并树剪枝方法，进行跨样本 Skill 更新。首先，我们以题目类型和修正类型构成 Patch 的语义签名，并在类型约束下将样本级 Patch 聚合为簇级编辑。其次，我们构造 Patch 合并树，将簇级编辑作为叶节点，通过LLM生成逐渐合并的树结构。验证集会自顶向下评估根节点及其子节点，并剪枝有害节点。我们将在 \textbf{{[}待补充：任务与基准{]}} 上，使用 \textbf{{[}待补充：执行模型与优化器模型{]}} 对所提方法进行系统评估，重点考察其任务性能、跨样本泛化。
\end{abstract}

\begin{figure*}[t]
\centering
\includegraphics[width=0.86\textwidth]{patchtree_method.pdf}
\caption{PatchTree 总体框架。方法首先从多次执行轨迹中提取带有题目类型、修正类型、目标位置和最小编辑的样本级 Patch；随后在类型兼容性与目标簇粒度约束下形成簇级 Patch，并在 epoch 末重新聚合长尾记录。簇级 Patch 作为叶节点构成合并树，每个内部节点保存可执行的条件化合并编辑。验证阶段优先评估根节点；若根节点未通过，则仅回退到其直接子节点，并对保留子节点的组合进行整体复验。}
\label{fig:patchtree-overview}
\end{figure*}

\hypertarget{introduction}{%
\section{Introduction}\label{introduction}}

大语言模型正在从通用文本生成器发展为能够调用工具、操作文件并完成长程任务的智能体。在这一过程中，模型能力不仅取决于参数中存储的知识，还取决于推理阶段可获得的程序性指导：如何分解任务、何时调用工具以及在何种条件下终止或回退。Agent Skill 为这类程序性知识提供了自然载体。一个 Skill 通常由自然语言指令、操作规范、辅助脚本和参考材料构成，并以外部文档的形式注入 Agent 的执行上下文。由于其内容可以被直接阅读、编辑和版本化，Skill 为冻结模型提供了一种轻量而透明的领域适配接口。

高质量 Skill 的构建仍然需要大量专业知识和反复调试。人工编写的规则难以预先覆盖真实任务中的输入变化与失败模式，而一次性生成的 Skill 又容易包含未经验证的假设。随着 Agent 在任务分布上持续执行，其轨迹、最终输出和验证器反馈提供了更直接的改进证据。自然语言反馈与文本空间优化研究已经表明，执行结果可以被转化为反思、文本梯度或局部编辑，并用于更新模型之外的语言状态 {[}1--5{]}。近期的 Skill 学习方法进一步从多条轨迹中抽取可复用经验，自动生成或迭代修订 Skill 文档 {[}6,8--10{]}。这些进展使反馈驱动的 Skill 优化成为一种可行的持续适配途径。

然而，从轨迹中生成局部修改建议并不等同于得到可靠的 Skill 更新。执行反馈天然对应具体样本，由此产生的 Patch 通常围绕一次局部失败。Skill 需要将这些局部修复转化为具有明确适用边界的通用规则。若按样本到达顺序逐条写回，Skill 会不断积累重复、细碎甚至相互矛盾的规则；若直接对整个 Patch 集合进行一次性总结，又可能把仅在不同条件下成立的修复合并为无条件指令，从而引入负迁移。因而，Skill 优化的关键不只在于产生 Patch，还在于组织 Patch 之间的关系。

现有研究已经探索了多种经验聚合方式。例如，轨迹级 lesson 可以通过层级归并压缩为统一的操作规范 {[}6{]}，任务样本可以按照 facet 或语义相似性组织后再进行局部优化 {[}7{]}，成功与失败轨迹也可以分别归纳为候选 Skill 并通过干预式评估进行筛选 {[}8{]}。这些方法说明跨样本证据聚合对于 Skill 构建至关重要，但聚合过程中的两个核心决策仍常被共同交给一次总结或固定归并流程隐式完成。第一个决策是\textbf{融合对象选择}：两个 Patch 是否具有兼容的任务条件和修复目标，因而能够形成同一条规则。第二个决策是\textbf{融合尺度选择}：多个局部规则能否继续抽象为覆盖范围更大的高层规则。前者决定融合是否正确，后者决定融合应在何处停止。

这两个决策存在不同的错误模式。仅依据输入主题或表面语义相似性聚类，可能把面对相似问题但修复意图不同的 Patch 放入同一组；仅依据修复动作聚类，又可能忽略规则触发条件的差异。即使一组 Patch 在局部层面兼容，将所有簇继续合并为单一全局编辑也未必最优。更高层融合能够减少冗余并扩大规则覆盖范围，但同时增加边界丢失和行为干扰的风险。因此，一个可靠的聚合方法既需要显式描述 Patch 的兼容关系，也需要保留不同融合层级的候选结果，使验证信号能够决定最终抽象尺度。

为此，本文提出 \textbf{PatchTree}，一种类型引导的簇级 Patch 聚合与合并树剪枝方法。对于当前 Skill 下失败或表现不稳定的训练样本，优化器联合分析多次执行轨迹，并生成结构化 Patch 记录

\[
R_i=(\alpha_i,\beta_i,\ell_i,p_i),
\]

其中 \(\alpha_i\) 表示题目类型，描述输入要求 Agent 完成的结构性行为；\(\beta_i\) 表示修正类型，描述 Patch 所针对的功能性缺口；\(\ell_i\) 和 \(p_i\) 分别表示目标 Skill 位置与最小编辑内容。题目类型和修正类型共同构成 Patch 的语义签名，使融合兼容性从最终总结器中的隐式判断转化为显式的中间表示。

方法的第一阶段执行类型引导的簇级 Patch 聚合。聚类器以题目类型和修正类型的兼容性为主要依据，并结合当前 Patch 数量控制簇粒度。同一簇中的样本级 Patch 被压缩为一个具有明确适用条件和行为边界的簇级编辑。簇级编辑由多个局部样本共同支持，构成跨样本写回的最低层单元。对于单个 batch 中暂时缺少共同支持的长尾记录，方法在 epoch 结束时进行统一再聚合，以发现分散在不同 batch 中的低频重复模式。

第二阶段构造 Patch 合并树。树的叶节点对应簇级编辑，内部节点保存其子节点进一步融合后形成的实际编辑，根节点表示本轮覆盖范围最大的候选更新。由此，树不只是组织 Patch 的数据结构，还显式表示从局部规则到高层规则的多个抽象尺度。验证集驱动的剪枝从根节点开始：若根候选提升验证性能，则接受完整融合；否则回退到根的直接子节点，保留具有正向收益的局部更新，并对其组合进行整体复验。验证信号因而同时决定候选是否写回以及写回应停留在哪个融合层级。

从整体上看，本文将跨样本 Skill 优化表述为一个结构化聚合问题。类型签名与簇级融合解决``哪些 Patch 可以共享同一规则''，Patch 合并树与验证剪枝解决``这些规则可以进一步抽象到何种范围''。这种分解保留了局部反馈的适用边界，同时为跨样本共性提供了逐层归纳的空间。

本文的主要贡献如下：

\begin{enumerate}
\def\labelenumi{\arabic{enumi}.}
\tightlist
\item
  我们提出\textbf{类型引导的簇级 Patch 聚合}，使用题目类型和修正类型联合刻画局部编辑的融合兼容性，并将相容的样本级 Patch 归纳为具有跨样本支持的簇级更新。
\item
  我们提出\textbf{Patch 合并树与验证集剪枝}，将逐层融合产生的实际编辑组织为结构化候选空间，并通过根节点优先、直接子节点回退的验证策略选择最终写回尺度。
\item
  我们构建完整的反馈驱动 Skill 优化系统，并将在 \textbf{{[}待补充：基准数量{]}} 个基准、\textbf{{[}待补充：模型数量{]}} 个执行模型与 \textbf{{[}待补充：环境数量{]}} 类 Agent 环境上评估其性能、泛化性、稳定性和成本。
\end{enumerate}

\hypertarget{related-work}{%
\section{Related Work}\label{related-work}}

\hypertarget{ux81eaux7136ux8bedux8a00ux53cdux9988ux4e0eux6587ux672cux7a7aux95f4ux4f18ux5316}{%
\subsection{自然语言反馈与文本空间优化}\label{ux81eaux7136ux8bedux8a00ux53cdux9988ux4e0eux6587ux672cux7a7aux95f4ux4f18ux5316}}

利用自然语言反馈改进模型行为，是反馈驱动 Skill 优化的基础。Reflexion 将环境反馈总结为可跨试次复用的语言记忆，使 Agent 能够在后续执行中规避已经观察到的失败 {[}1{]}。Prompt 优化研究进一步把语言指令本身视为可搜索变量。Automatic Prompt Engineer 通过候选生成与评分自动发现任务指令 {[}11{]}；ProTeGi 根据小批量错误产生文本梯度，并结合 beam search 和 bandit selection 迭代 Prompt {[}2{]}；OPRO 让 LLM 根据历史候选及其分数提出新的优化解 {[}3{]}；DSPy 将多阶段语言程序声明为可编译模块，并联合优化指令和示例 {[}12{]}。

近期方法开始利用更丰富的结构化反馈。TextGrad 将语言系统表示为文本计算图，使自然语言反馈可以沿组件依赖关系传播到不同文本变量 {[}4{]}；Task Facet Learning 通过发现任务子结构，将全局 Prompt 重写转化为面向特定 facet 的局部修改 {[}7{]}；GEPA 从完整执行轨迹中生成反思，并通过演化选择保留在多个样本上有效的语言程序 {[}5{]}。这些工作主要研究如何生成和搜索高质量文本更新。本文研究其后的聚合问题：当多个样本已经产生局部 Patch 时，如何根据适用条件和修复目标组织这些编辑，并确定适合写回 Skill 的抽象范围。

\hypertarget{agent-skill-ux7684ux6784ux5efaux4e0eux6f14ux5316}{%
\subsection{Agent Skill 的构建与演化}\label{agent-skill-ux7684ux6784ux5efaux4e0eux6f14ux5316}}

Agent Skill 将可复用程序性知识保存在模型参数之外。Voyager 在开放世界环境中维护可检索和组合的代码 Skill 库 {[}13{]}，ExpeL 从跨任务轨迹中抽取经验规则以指导后续执行 {[}14{]}。这些工作展示了外部经验状态在长期 Agent 学习中的价值。近期研究进一步系统化了 Skill 的定义、生命周期和评测。Agentic Skills SoK 从适用条件、执行过程、工具协调与治理方式等维度梳理了 Skill 的组成 {[}15{]}；SkillsBench 通过成对任务评测考察人工和自动生成 Skill 的实际收益，表明 Skill 的质量与适用边界直接影响其能否带来稳定改进 {[}16{]}。

从执行轨迹中归纳 Skill 已成为重要研究方向。AutoSkill 在冻结执行模型的条件下持续发现和验证可复用规则 {[}17{]}；Trace2Skill 从单条成功或失败轨迹中并行抽取 trajectory-local lesson，再通过层级聚合形成紧凑的 Skill 目录 {[}6{]}；CoEvoSkills 通过 Skill Generator 与独立 Surrogate Verifier 的协同过程迭代构建多文件 Skill package {[}18{]}；SkillGen 根据历史轨迹合成候选 Skill，并利用同任务的配对干预评估其净收益 {[}8{]}。这些方法从不同角度处理了经验提取、冲突消解和候选验证问题，证明多轨迹证据能够被转化为可部署的程序性知识。

本文关注持续 Skill 优化中的跨样本编辑聚合。其基本单元不是完整轨迹或独立 Skill，而是由轨迹反馈产生的最小 Patch。通过同时建模题目结构与修复意图，本文在写回前显式刻画 Patch 间的兼容关系，并将聚类结果直接转化为可执行的簇级编辑。

\hypertarget{ux53cdux9988ux9a71ux52a8ux7684-skill-ux4f18ux5316ux4e0eux8de8ux6837ux672cux805aux5408}{%
\subsection{反馈驱动的 Skill 优化与跨样本聚合}\label{ux53cdux9988ux9a71ux52a8ux7684-skill-ux4f18ux5316ux4e0eux8de8ux6837ux672cux805aux5408}}

反馈驱动的 Skill 文本优化将 Skill 文档视为可随执行经验更新的外部状态。SkillOpt 使用独立优化器将 scored rollouts 转化为结构化增删改操作，并通过 held-out validation gate 选择候选 Skill {[}9{]}。SkillGrad 将轨迹诊断表示为 textual gradient，通过持续积累的模式信息指导分层 Skill package 更新 {[}19{]}。SkillOpt-Lite 则以轨迹探索、共识属性挖掘和独立验证为核心，展示了简洁更新流程在 Skill 自进化中的有效性 {[}10{]}。这些研究确立了从执行证据产生候选编辑、再以任务表现决定写回的基本闭环。

与本文最相关的是跨样本聚合机制。Task Facet Learning 根据任务子结构组织样本，使局部反馈作用于对应的 Prompt section {[}7{]}。Trace2Skill 先从单条轨迹提取 lesson，再通过多层 consolidation 消除重复和冲突 {[}6{]}。SkillComposer 将 Create、Merge 和 Improve 建模为 Skill 演化操作，并以 source-task 表现约束 Skill 合并 {[}20{]}。SkillGen 则强调候选 Skill 作为外部干预所带来的净收益 {[}8{]}。这些方法分别从任务分解、层级归纳、Skill 库演化和因果验证角度处理经验融合。

现有聚合过程通常以主题相似性、任务 facet 或完整 Skill 为组织单位，而局部 Patch 是否共享相同触发条件与修复目标仍多由归纳模型隐式判断。此外，层级聚合常被用于生成一个最终结果，树中间的融合状态较少作为可独立验证和回退的候选。本文以``题目类型---修正类型''签名显式描述 Patch 兼容性，并保留合并树各层的实际编辑，使独立验证信号能够选择最终融合尺度。

\hypertarget{method}{%
\section{Method}\label{method}}

\hypertarget{ux95eeux9898ux5b9aux4e49}{%
\subsection{问题定义与总体框架}\label{ux95eeux9898ux5b9aux4e49}}

给定冻结执行模型 \(\pi\)、优化器模型 \(\mathcal O\)、当前 Skill \(S_t\)、训练集 \(\mathcal D_{\mathrm{train}}\)、验证集 \(\mathcal D_{\mathrm{val}}\)、测试集 \(\mathcal D_{\mathrm{test}}\)，以及任务奖励函数 \(r(x,\tau)\in[0,1]\)，本文目标是在不更新执行模型参数的条件下，根据训练任务上的执行证据优化 Skill：

\[
S_{t+1}=\operatorname{Optimize}(S_t,\pi,\mathcal O,\mathcal D_{\mathrm{train}},\mathcal D_{\mathrm{val}}).
\]

对于输入 \(x\) 和 Skill \(S\)，执行模型产生轨迹

\[
\tau\sim\pi(\cdot\mid x,S),
\]

任务验证器根据最终状态、答案或环境结果返回奖励 \(r(x,\tau)\)。记 \(S\oplus e\) 为将结构化编辑 \(e\) 应用于 Skill \(S\) 后得到的新版本。编辑可以作用于 Skill 的既有段落，也可以增加具有明确触发条件的新规则。

定义 Skill 在数据集 \(\mathcal D\) 上的期望表现为

\[
V_{\mathcal D}(S)=\frac{1}{|\mathcal D|}\sum_{x\in\mathcal D}\mathbb E_{\tau\sim\pi(\cdot\mid x,S)}[r(x,\tau)].
\]

训练集负责产生执行轨迹、样本级 Patch 和合并树候选；验证集负责选择写回层级并决定是否接受更新；测试集仅在整个优化过程结束后用于最终评估。该划分使生成候选的样本与决定候选泛化价值的样本相互分离。

本文关注局部 Patch 到全局 Skill 更新之间的两个决策。给定一组样本级 Patch，首先需要确定可以共享同一规则的兼容子集；在得到多个簇级规则后，还需要确定它们能够继续融合的抽象范围。形式上，优化过程由

\[
\underbrace{\operatorname{Cluster}(\{p_i\})}_{\text{融合对象选择}}
\qquad\text{和}\qquad
\underbrace{\operatorname{Prune}(T)}_{\text{融合尺度选择}}
\]

共同构成。

\paragraph{总体流程.}

在第 \(t\) 轮优化中，执行模型首先使用当前 Skill \(S_t\) 对训练样本进行多次独立 rollout。对于失败或表现不稳定的样本，Patch 分析器联合读取输入、执行轨迹和验证器反馈，生成由题目类型、修正类型、目标位置和最小编辑组成的结构化记录。

随后，类型聚类器根据题目类型与修正类型判断记录之间的兼容性，并结合当前记录数量控制目标簇粒度。每个多样本簇由簇内融合器归纳为一个簇级 Patch。簇级 Patch 是合并树的叶节点，也是方法中最细粒度的跨样本更新单元。

在此基础上，合并规划器根据各叶节点的类型原型、目标位置、适用条件和行为边界构造 Patch 合并树。内部节点保存直接子节点进一步融合后形成的实际编辑，根节点对应本轮覆盖范围最大的候选。最后，验证集剪枝从根节点开始选择写回结果；当根节点未能改善验证性能时，算法回退到根的直接子节点，并验证具有正向收益的子节点组合。

整体流程可以写为

\[
\begin{aligned}
\text{Rollout}&\rightarrow\text{样本级 Patch}\rightarrow\text{类型聚类}\\
&\rightarrow\text{簇级融合}\rightarrow\text{Patch 合并树}\rightarrow\text{验证剪枝}.
\end{aligned}
\]

\hypertarget{ux591aux91c7ux6837ux6837ux672cux7ea7-patch-ux751fux6210}{%
\subsection{类型引导的簇级 Patch 聚合}
\subsubsection{多采样样本级 Patch 生成}\label{ux591aux91c7ux6837ux6837ux672cux7ea7-patch-ux751fux6210}}

对于训练样本 \(x_i\in\mathcal D_{\mathrm{train}}\)，在当前 Skill 下独立采样 \(K\) 条执行轨迹：

\[
\tau_{ik}\sim\pi(\cdot\mid x_i,S_t),\qquad k=1,\ldots,K.
\]

样本的经验成功率为

\[
q_i(S_t)=\frac{1}{K}\sum_{k=1}^{K}r(x_i,\tau_{ik}).
\]

当 \(q_i(S_t)\ge\tau_{\mathrm{succ}}\) 时，当前 Skill 已能稳定处理该样本，本轮不再为其生成修复编辑。对于 \(q_i(S_t)<\tau_{\mathrm{succ}}\) 的样本，Patch 分析器综合多条轨迹中反复出现的错误、成功片段与验证器反馈，输出

\[
R_i=(\alpha_i,\beta_i,\ell_i,p_i).
\]

其中，题目类型 \(\alpha_i\) 描述输入对 Agent 行为提出的结构性需求，例如显式约束遵循、多步依赖推理、证据支持回答、格式受控生成、比较与选择或工具调用决策。题目类型强调求解结构，而不是数学、代码或写作等宽泛领域标签。

修正类型 \(\beta_i\) 描述 Patch 的功能性修复意图，例如约束核验、步骤分解、证据检查、格式强制、计算复核、歧义澄清或过度泛化控制。修正类型强调需要补充或约束的行为机制，而不是 add、delete 或 replace 等表面编辑动作。

目标位置 \(\ell_i\) 指明编辑应作用于 Skill 的哪一部分，例如规划规则、工具策略、验证步骤或输出规范。\(p_i\) 是可执行的最小 Patch，要求表达一项清晰、可复用的行为修改。用于聚类的核心签名为

\[
s_i=(\alpha_i,\beta_i).
\]

签名需要保持简洁、稳定和去实例化，不包含原题中的实体、数值或标准答案。目标位置与编辑内容用于后续融合约束，但不替代题目类型和修正类型对语义兼容性的描述。一个样本记录可表示为：

\begin{verbatim}
question_type: explicit-constraint following
revision_type: constraint-verification
target: verification rules
patch: require every explicit constraint to be checked before producing the final answer
\end{verbatim}

\hypertarget{ux7c7bux578bux4e0eux6570ux91cfux611fux77e5ux7684-patch-ux805aux7c7b}{%
\subsubsection{类型与数量感知的 Patch 聚类}\label{ux7c7bux578bux4e0eux6570ux91cfux611fux77e5ux7684-patch-ux805aux7c7b}}

收集当前 batch 中所有待修复样本的 Patch 记录：

\[
\mathcal R_t=\{R_i:q_i(S_t)<\tau_{\mathrm{succ}}\},\qquad B_t=|\mathcal R_t|.
\]

若 \(B_t=0\)，当前 batch 不产生候选更新。否则，设期望簇大小为 \(s_{\mathrm{cluster}}\)，目标簇数为

\[
M_{\mathrm{target}}=\max\left(1,\operatorname{round}\frac{B_t}{s_{\mathrm{cluster}}}\right).
\]

当前实现取 \(s_{\mathrm{cluster}}=7\)，使每个簇通常包含约 6-\/-8 个记录。\(M_{\mathrm{target}}\) 为聚类器提供粒度先验，而不是强制均匀划分。类型兼容性始终具有更高优先级：当两个记录的结构性需求或修正方向明显不兼容时，即使合并能够使簇大小更接近目标值，也应保持分离。

为降低聚类输入长度，首先将完全相同的类型签名压缩为

\[
\widehat{\mathcal S}_t=\{(\widehat s_m,n_m,I_m)\}_{m=1}^{N_t},
\]

其中 \(\widehat s_m\) 是唯一签名，\(n_m\) 是对应记录数，\(I_m\) 是记录索引集合。LLM 聚类器读取类型签名、数量、目标位置与简短 Patch 摘要，输出

\[
\mathcal C_t=\{C_1,\ldots,C_M\}.
\]

每个簇具有类型原型

\[
P_j=(A_j,B_j),
\]

其中 \(A_j\) 概括簇内共享的题目结构，\(B_j\) 概括共享的修正方向。一个有效簇对应一类 \emph{type-compatible repair}：成员可以来自不同领域或具有不同表面表达，但应当能够由同一条带条件的 Skill 规则处理。

\hypertarget{ux7c07ux7ea7-patch-ux878dux5408ux4e0eux957fux5c3eux8bb0ux5f55ux5904ux7406}{%
\subsubsection{簇级 Patch 融合与长尾记录处理}\label{ux7c07ux7ea7-patch-ux878dux5408ux4e0eux957fux5c3eux8bb0ux5f55ux5904ux7406}}

对于包含至少两个记录的簇 \(C_j\)，簇内融合器根据类型原型、目标位置和成员 Patch 生成簇级编辑

\[
e_j=\operatorname{MergeLeaf}\left(P_j,\{(\ell_i,p_i):R_i\in C_j\}\right).
\]

簇级融合遵循三个原则。首先，\(e_j\) 应覆盖簇内记录共同支持的修复机制，成员中仅出现一次的实例细节不进入共享规则。其次，编辑需要显式保留适用条件和不适用边界，避免将条件化行为改写为无条件约束。最后，当成员 Patch 指向不同 Skill 位置时，只有在行为目标和执行顺序兼容的情况下才形成复合编辑；否则应在聚类阶段保持分离。

由此得到的簇级 Patch 构成最低层的跨样本更新。每个叶节点表示为

\[
u_j=(e_j,P_j,C_j),
\]

并保留支持样本数量、目标位置、适用条件和编辑摘要，供后续合并规划使用。

若当前 batch 中某个簇仅包含一个记录，则将其暂存到当前 epoch 的长尾缓冲区 \(\mathcal B_{\mathrm{tail}}\)。在 epoch 结束时，来自不同 batch 的长尾记录被统一重新聚类：

\[
\mathcal C_{\mathrm{tail}}=\operatorname{Cluster}(\mathcal B_{\mathrm{tail}}).
\]

其中形成的多样本簇按照相同规则生成簇级 Patch；仍然缺少跨样本支持的记录在该 epoch 结束后移出候选集合。长尾处理改变的是聚合时机，使分散出现的低频模式能够在更大的时间窗口中获得共同支持。

最终叶节点集合为

\[
\mathcal U_t=\{u_j=(e_j,P_j,C_j):|C_j|\ge2\}.
\]

当 \(\mathcal U_t=\varnothing\) 时，本轮没有可供跨样本写回的候选，保持 \(S_{t+1}=S_t\)。

\hypertarget{patch-ux5408ux5e76ux6811}{%
\subsection{Patch 合并树构建与剪枝}
\subsubsection{Patch 合并树构建}\label{patch-ux5408ux5e76ux6811}}

给定叶节点集合 \(\mathcal U_t\)，合并规划器构造一棵有界 Patch 合并树

\[
T_t=(\mathcal V_t,\mathcal E_t).
\]

每个叶节点以节点卡片的形式输入规划器，卡片包含簇级题目类型、修正类型、目标位置、编辑摘要、适用条件、行为边界和支持样本数。规划器根据这些信息决定哪些节点具有共同的上位行为目标、哪些节点应保持在不同分支，以及逐层融合后应形成怎样的编辑。

对于内部节点 \(v\)，其编辑由直接子节点融合得到：

\[
e_v=\operatorname{MergeNode}\left(\{e_c:c\in\operatorname{child}(v)\}\right).
\]

内部节点同时保存

\[
m_v=(\text{merge intent},\text{applicability},\text{boundary}),
\]

分别描述子节点共享的上位目标、合并规则的触发条件以及必须保持分离的行为边界。\(e_v\) 是能够直接作用于当前 Skill 的实际编辑，而不仅是供上一层使用的摘要。因而，树中每一层都对应一个可执行的融合尺度：靠近叶节点的编辑保留更具体的适用条件，靠近根节点的编辑覆盖更多修复模式并具有更高抽象程度。

根节点 \(r\) 对应本轮最大范围的融合候选：

\[
\widetilde S_t^{\mathrm{root}}=S_t\oplus e_r.
\]

规划器允许不兼容节点保持为不同分支，从而避免为了形成单一规则而丢失行为边界。为控制候选规模，每个内部节点的直接子节点数不超过 \(b\)，树深度不超过 \(H_{\max}\)。若叶节点集合只包含一个节点，该节点同时作为根节点。

Patch 合并树将融合尺度表示为具有父子关系的候选空间。与枚举任意 Patch 子集相比，该结构利用融合过程自身的层次关系提供自然回退路径，使高层候选失效时能够定位到更保守的直接子结构。

\hypertarget{ux9a8cux8bc1ux96c6ux9a71ux52a8ux7684ux81eaux9876ux5411ux4e0bux526aux679d}{%
\subsubsection{验证集驱动的自顶向下剪枝}\label{ux9a8cux8bc1ux96c6ux9a71ux52a8ux7684ux81eaux9876ux5411ux4e0bux526aux679d}}

合并树构造完成后，首先评估根节点候选。当前 Skill 与候选 Skill 使用相同的验证样本和采样预算，以降低任务差异和随机执行噪声。根节点的验证收益为

\[
\Delta_{\mathrm{root}}^{\mathrm{val}}=V_{\mathcal D_{\mathrm{val}}}(\widetilde S_t^{\mathrm{root}})-V_{\mathcal D_{\mathrm{val}}}(S_t).
\]

当

\[
\Delta_{\mathrm{root}}^{\mathrm{val}}>\tau_{\mathrm{val}}
\]

时，接受最大范围融合并令 \(S_{t+1}=\widetilde S_t^{\mathrm{root}}\)。

若根节点未通过验证，算法回退到其直接子节点 \(\operatorname{child}(r)=\{v_1,\ldots,v_m\}\)，分别计算

\[
\Delta_{v_j}^{\mathrm{val}}=V_{\mathcal D_{\mathrm{val}}}(S_t\oplus e_{v_j})-V_{\mathcal D_{\mathrm{val}}}(S_t).
\]

保留验证收益超过子节点阈值的节点：

\[
\mathcal C^+=\{v_j:\Delta_{v_j}^{\mathrm{val}}>\tau_{\mathrm{child}}\}.
\]

若 \(\mathcal C^+=\varnothing\)，本轮候选无法提供可靠改进，保持当前 Skill。否则，将保留节点作为具有独立适用条件的编辑共同应用：

\[
\widetilde S_t^{\mathrm{child}}=S_t\oplus\{e_v:v\in\mathcal C^+\}.
\]

当多个子编辑作用于相同位置时，写回器按照树中记录的依赖顺序插入规则，并保留各自的触发条件。组合候选需要再次进行整体验证：

\[
V_{\mathcal D_{\mathrm{val}}}(\widetilde S_t^{\mathrm{child}})-V_{\mathcal D_{\mathrm{val}}}(S_t)>\tau_{\mathrm{val}}.
\]

若组合候选通过验证，则令 \(S_{t+1}=\widetilde S_t^{\mathrm{child}}\)；否则保持 \(S_{t+1}=S_t\)。整体复验用于识别多个单独有效编辑共同写入时可能产生的交互退化。

剪枝过程只从根节点回退到其直接子节点，不进一步递归搜索更深层节点，也不枚举任意节点组合。该策略把每轮验证候选数控制在 \(O(b)\)，同时保留一次从高抽象规则回退到较局部规则的机会。验证集由此承担两个作用：判断本轮更新是否具有泛化收益，以及在候选树中选择合适的融合尺度。

\hypertarget{ux5b8cux6574ux4f18ux5316ux8fc7ux7a0b}{%
\subsubsection{完整优化过程}\label{ux5b8cux6574ux4f18ux5316ux8fc7ux7a0b}}

\begin{verbatim}
Algorithm 1: Type-Guided Cluster-Level Patch Aggregation with Merge-Tree Pruning

Input:
    current Skill S_t; frozen executor π; optimizer O
    D_train and D_val; rollout count K
    thresholds τ_succ, τ_val, τ_child
    target cluster size s_cluster
    branching bound b and depth bound H_max

1:  R_t ← ∅
2:  for each x_i in the current training batch do
3:      sample K trajectories with (π, S_t)
4:      compute q_i(S_t)
5:      if q_i(S_t) < τ_succ then
6:          generate R_i = (α_i, β_i, ℓ_i, p_i)
7:          R_t ← R_t ∪ {R_i}
8:      end if
9:  end for
10: if R_t = ∅ then return S_t
11: M_target ← max(1, round(|R_t| / s_cluster))
12: C_t ← TypeCluster(R_t, M_target)
13: U_t ← ∅
14: for each cluster C_j in C_t do
15:     if |C_j| = 1 then
16:         add C_j to the epoch tail buffer
17:     else
18:         e_j ← MergeLeaf({(ℓ_i, p_i): R_i ∈ C_j})
19:         U_t ← U_t ∪ {(e_j, P_j, C_j)}
20:     end if
21: end for
22: if U_t = ∅ then return S_t
23: T_t ← PlanMergeTree(U_t; b, H_max)
24: evaluate the root candidate on D_val
25: if root gain > τ_val then return root candidate
26: evaluate each direct child of the root on D_val
27: C+ ← children with gain > τ_child
28: if C+ = ∅ then return S_t
29: combine C+ as condition-preserving edits
30: if combined gain > τ_val then return combined candidate
31: return S_t

At the end of each epoch:
32: re-cluster all records in the tail buffer
33: build leaves from newly formed multi-sample clusters
34: run Lines 23–31 and remove records that remain singleton
\end{verbatim}
\begin{thebibliography}{99}
\small

\bibitem{reflexion}
Shinn et al.
\newblock Reflexion: Language Agents with Verbal Reinforcement Learning.
\newblock \emph{NeurIPS}, 2023.

\bibitem{protegi}
Pryzant et al.
\newblock Automatic Prompt Optimization with ``Gradient Descent'' and Beam Search.
\newblock \emph{EMNLP}, 2023.

\bibitem{opro}
Yang et al.
\newblock Large Language Models as Optimizers.
\newblock \emph{ICLR}, 2024.

\bibitem{textgrad}
Yuksekgönül et al.
\newblock TextGrad: Automatic ``Differentiation'' via Text.
\newblock arXiv:2406.07496, 2024.

\bibitem{gepa}
Agrawal et al.
\newblock GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning.
\newblock \emph{ICLR}, 2026.

\bibitem{trace2skill}
\newblock Trace2Skill: Distill Trajectory-Local Lessons into Transferable Agent Skills.
\newblock arXiv:2603.25158, 2026.

\bibitem{taskfacet}
\newblock Task Facet Learning: A Structured Approach to Prompt Optimization.
\newblock arXiv:2406.10504, 2024.

\bibitem{skillgen}
\newblock SkillGen: Verified Inference-Time Agent Skill Synthesis.
\newblock arXiv:2605.10999, 2026.

\bibitem{skillopt}
Yang et al.
\newblock SkillOpt: Executive Strategy for Self-Evolving Agent Skills.
\newblock arXiv:2605.23904, 2026.

\bibitem{skilloptlite}
\newblock SkillOpt-Lite: Better and Faster Agent Self-evolution via One Line of Vibe.
\newblock arXiv:2607.03451, 2026.

\bibitem{ape}
Zhou et al.
\newblock Large Language Models Are Human-Level Prompt Engineers.
\newblock \emph{ICLR}, 2023.

\bibitem{dspy}
Khattab et al.
\newblock DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines.
\newblock \emph{ICLR}, 2024.

\bibitem{voyager}
Wang et al.
\newblock Voyager: An Open-Ended Embodied Agent with Large Language Models.
\newblock 2023.

\bibitem{expel}
Zhao et al.
\newblock ExpeL: LLM Agents Are Experiential Learners.
\newblock \emph{AAAI}, 2024.

\bibitem{skills-sok}
\newblock SoK: Agentic Skills---Beyond Tool Use in LLM Agents.
\newblock arXiv:2602.20867, 2026.

\bibitem{skillsbench}
\newblock SkillsBench: Benchmarking How Well Agent Skills Work Across Diverse Tasks.
\newblock arXiv:2602.12670, 2026.

\bibitem{autoskill}
\newblock AutoSkill: Towards Automated and Dynamic Skill Discovery for Agents.
\newblock arXiv:2603.01145, 2026.

\bibitem{coevoskills}
\newblock CoEvoSkills: Self-Evolving Agent Skills via Co-Evolutionary Verification.
\newblock arXiv:2604.01687, 2026.

\bibitem{skillgrad}
\newblock SkillGrad: Optimizing Agent Skills Like Gradient Descent.
\newblock arXiv:2605.27760, 2026.

\bibitem{skillcomposer}
\newblock SkillComposer: Learning to Evolve Agent Skills for Specification and Generalization.
\newblock arXiv:2606.06079, 2026.

\end{thebibliography}

\end{document}
