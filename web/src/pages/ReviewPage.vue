<script setup>
import { ref, reactive } from 'vue'
import { marked } from 'marked'

defineProps({
  activeMode: { type: String, default: 'review' },
})
const emit = defineEmits(['switch-mode'])

// ============ 状态机 ============
// input(初始) → keywords(确认关键词) → searching → papers(看召回) → running(生成中) → result / error
const phase = ref('input')
const userQuery = ref('多标签学习')      // 用户原问题（贯穿各卡点回显）
const errorMsg = ref('')

let eventSource = null

// 卡点1：关键词
const keywords = reactive({ querys: [], start_date: null, end_date: null })
const editableKeywords = ref([])

// 卡点2：召回论文
const recall = reactive({ count: 0, papers: [], querys: [], few: false })

// 生成进度时间线
const steps = ref([])
const STEP_DEFS = [
  { key: 'reading', title: '抽取与聚类' },
  { key: 'writing', title: '撰写综述' },
  { key: 'faithfulness_checking', title: '忠实度校验' },
]
const reviewMarkdown = ref('')
const faithReport = ref(null)            // 忠实度校验结果：{verdict,total_claims,revised,remaining_bad,detail[]}
const reviewWarning = ref('')

// ============ 启动 ============
function start() {
  if (!userQuery.value.trim()) return
  errorMsg.value = ''
  reviewMarkdown.value = ''
  reviewWarning.value = ''
  steps.value = []
  phase.value = 'connecting'
  eventSource = new EventSource(`/api/research?query=${encodeURIComponent(userQuery.value)}`)
  eventSource.onmessage = (e) => {
    try { handle(JSON.parse(e.data)) } catch (err) { console.error(err) }
  }
  eventSource.onerror = () => { fail('与服务器的连接中断') }
}

// ============ SSE 分发 ============
function handle({ step, state, data }) {
  // 前端埋点：进度类逐条打印（generating 太碎，只打首次）；出问题时复制 console 给后端排查
  if (state !== 'generating') {
    console.log(`[SSE] step=${step} state=${state}`, typeof data === 'string' ? data.slice(0, 100) : data)
  }
  switch (state) {
    case 'user_review': return onKeywordReview(data)
    case 'papers_review': return onPapersReview(data)
    case 'initializing':
    case 'thinking': return onProgress(step, state, data)
    case 'generating': return onGenerating(step, data)
    case 'completed': return onCompleted(step, data)
    case 'error': return fail(data)
    case 'finished': return onFinished()
  }
}

function onKeywordReview(raw) {
  try {
    const obj = JSON.parse(raw)
    keywords.querys = obj.querys || []
    keywords.start_date = obj.start_date || null
    keywords.end_date = obj.end_date || null
  } catch { keywords.querys = [] }
  editableKeywords.value = [...keywords.querys]
  phase.value = 'keywords'
}

function confirmKeywords() {
  const payload = JSON.stringify({
    querys: editableKeywords.value.map(s => s.trim()).filter(Boolean),
    start_date: keywords.start_date,
    end_date: keywords.end_date,
  })
  sendInput(payload)
  phase.value = 'searching'
}
function addKeyword() { editableKeywords.value.push('') }
function removeKeyword(i) { editableKeywords.value.splice(i, 1) }

function onPapersReview(raw) {
  const obj = JSON.parse(raw)
  recall.count = obj.count
  recall.papers = obj.papers || []
  recall.querys = obj.querys || []
  recall.few = obj.few
  if (obj.user_request) userQuery.value = obj.user_request
  phase.value = 'papers'
}
function confirmPapers() { sendInput('continue'); enterRunning() }
function abortPapers() { sendInput('abort') }

function enterRunning() {
  phase.value = 'running'
  steps.value = STEP_DEFS.map(d => ({ ...d, detail: '', status: 'wait' }))
}

function onProgress(step, state, data) {
  if (phase.value !== 'running') enterRunning()
  const s = findStep(step)
  if (s) {
    s.status = 'run'
    // 撰写综述阶段：正文会在下方完整显示，进度区只显示"进行中"动画，不再回显大段文字
    if (s.key !== 'writing' && typeof data === 'string' && data.trim()) {
      s.detail = data.trim()
    }
  }
}

function onGenerating(step, data) {
  if (step === 'writing' && typeof data === 'string') {
    reviewMarkdown.value += data
    const s = findStep('writing'); if (s) s.status = 'run'
  }
}

function onCompleted(step, data) {
  const s = findStep(step)
  if (s) {
    s.status = 'done'
    // writing 阶段不回显长正文；其余阶段显示简短汇总
    if (s.key !== 'writing' && s.key !== 'faithfulness_checking' && typeof data === 'string' && data.trim()) {
      s.detail = data.trim()
    }
  }
  if (step === 'reading' && typeof data === 'string' && data.includes('证据规模偏小')) {
    reviewWarning.value = data.trim()
  }
  if (step === 'writing' && typeof data === 'string' && data.length > reviewMarkdown.value.length) {
    reviewMarkdown.value = data
  }
  if (step === 'faithfulness_checking') {
    // 校验结果是结构化 JSON：汇总 + 仍不达标详情
    try {
      const r = JSON.parse(data)
      faithReport.value = r
      if (s) s.detail = `${r.verdict === 'pass' ? '全部通过' : '带警告通过'} · 共${r.total_claims}条论断，修订${r.revised}条，仍不达标${r.remaining_bad}条`
    } catch {
      if (s) s.detail = typeof data === 'string' ? data : '校验完成'
    }
    phase.value = 'result'
  }
}

function onFinished() {
  if (phase.value === 'running') phase.value = 'result'
  closeSSE()
}

function fail(msg) {
  errorMsg.value = typeof msg === 'string' ? msg : '生成失败'
  if (errorMsg.value.includes('修改关键词') || errorMsg.value.includes('召回')) {
    phase.value = 'keywords'
  } else if (['running', 'searching', 'connecting'].includes(phase.value)) {
    phase.value = 'error'
  }
  const running = steps.value.find(s => s.status === 'run')
  if (running) running.status = 'error'
  closeSSE()
}

function findStep(step) {
  const key = step && step.startsWith('section_writing') ? 'writing' : step
  return steps.value.find(s => s.key === key)
}
async function sendInput(input) {
  await fetch('/send_input', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ input }),
  })
}
async function loadLatestReport() {
  errorMsg.value = ''
  closeSSE()
  try {
    const response = await fetch('/api/research/latest_report')
    if (!response.ok) throw new Error('没有找到可展示的最新综述')
    const obj = await response.json()
    reviewMarkdown.value = obj.markdown || ''
    userQuery.value = obj.report?.user_request || userQuery.value
    faithReport.value = obj.report?.faithfulness_report || null
    reviewWarning.value = obj.report?.review_warning || ''
    steps.value = STEP_DEFS.map(d => ({ ...d, detail: '', status: 'done' }))
    phase.value = 'result'
  } catch (err) {
    fail(err.message || '加载最新综述失败')
  }
}
function closeSSE() { if (eventSource) { eventSource.close(); eventSource = null } }
function reset() {
  closeSSE(); phase.value = 'input'; errorMsg.value = ''
  reviewMarkdown.value = ''; steps.value = []; faithReport.value = null; reviewWarning.value = ''
}

// 下载综述为 Markdown 文件，文件名取综述正文里的一级标题
function downloadMd() {
  const m = reviewMarkdown.value.match(/^#\s+(.+)$/m)
  let title = m ? m[1].trim() : (userQuery.value || 'review').trim()
  title = title.replace(/[\\/:*?"<>|]/g, '').slice(0, 80)   // 去掉文件名非法字符
  const blob = new Blob([reviewMarkdown.value], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${title}.md`
  a.click()
  URL.revokeObjectURL(url)
}
function renderedReview() {
  const withSrc = reviewMarkdown.value.replace(
    /\[来源[:：]\s*([^\]]+)\]/g,
    '<span class="src">[来源: $1]</span>'
  )
  return marked.parse(withSrc)
}
</script>

<template>
  <div class="page">
    <header class="top">
      <div class="crest">P</div>
      <div class="name">Paper<span>Lens</span></div>
      <nav class="mode-tabs" aria-label="功能模式">
        <button class="active" type="button">综述分析</button>
        <button type="button" @click="emit('switch-mode', 'proposal')">研究方向推荐</button>
      </nav>
      <div class="tag">导航型文献综述</div>
    </header>

    <!-- ① 输入态 -->
    <section v-if="phase === 'input'" class="hero">
      <h1>把一个领域，读成一张<span class="u">研究地图</span></h1>
      <p class="lead">输入研究方向，自动检索 arXiv、聚类成研究脉络，生成每句可溯源的导航型综述。</p>
      <div class="field">
        <input v-model="userQuery" placeholder="例如：多标签学习、扩散模型、联邦学习…" @keyup.enter="start" />
        <button @click="start">生成综述</button>
      </div>
      <div class="actions center latest-actions">
        <button class="ghost" type="button" @click="loadLatestReport">查看最新报告</button>
      </div>
    </section>

    <!-- 连接 / 检索中 -->
    <section v-else-if="phase === 'connecting' || phase === 'searching'" class="wait-box">
      <div class="orig-q">研究主题：<b>{{ userQuery }}</b></div>
      <div class="spinner"></div>
      <p>{{ phase === 'connecting' ? '正在生成检索关键词…' : '正在检索 arXiv…' }}</p>
    </section>

    <!-- ② 卡点1：确认关键词 -->
    <section v-else-if="phase === 'keywords'" class="card">
      <div class="orig-q">研究主题：<b>{{ userQuery }}</b></div>
      <h2 class="card-title">确认检索关键词</h2>
      <p class="card-hint">系统据你的主题提取了以下英文关键词，可增删或修改后再检索。</p>
      <div v-if="errorMsg" class="err-inline">{{ errorMsg }}</div>
      <div class="kw-list">
        <div v-for="(kw, i) in editableKeywords" :key="i" class="kw-item">
          <input v-model="editableKeywords[i]" />
          <button class="kw-del" @click="removeKeyword(i)">×</button>
        </div>
        <button class="kw-add" @click="addKeyword">+ 添加关键词</button>
      </div>
      <div v-if="keywords.start_date || keywords.end_date" class="kw-date">
        时间范围：{{ keywords.start_date || '不限' }} ~ {{ keywords.end_date || '不限' }}
      </div>
      <div class="actions">
        <button class="ghost" @click="reset">返回修改主题</button>
        <button class="primary" @click="confirmKeywords">确认，开始检索</button>
      </div>
    </section>

    <!-- ③ 卡点2：看召回论文 -->
    <section v-else-if="phase === 'papers'" class="card">
      <div class="orig-q">研究主题：<b>{{ userQuery }}</b></div>
      <h2 class="card-title">召回论文确认</h2>
      <p class="card-hint">
        关键词 <code>{{ recall.querys.join(' / ') }}</code> 共召回 <b>{{ recall.count }}</b> 篇。
        请确认是否对路；若整体跑偏，可返回修改关键词。
      </p>
      <div v-if="recall.few" class="err-inline">召回偏少，可能领域较冷门或关键词偏窄。</div>
      <ul class="paper-list">
        <li v-for="(p, i) in recall.papers" :key="i">
          <span class="p-idx">{{ i + 1 }}</span>
          <span class="p-title">{{ p.title }}</span>
          <span class="p-year">{{ p.published }}</span>
        </li>
      </ul>
      <div class="actions">
        <button class="ghost" @click="phase = 'keywords'">返回改关键词</button>
        <button class="primary" @click="confirmPapers">看着对，继续生成</button>
      </div>
    </section>

    <!-- ④ 生成中 + ⑤ 结果 -->
    <section v-else-if="phase === 'running' || phase === 'result'">
      <div class="orig-q big">研究主题：<b>{{ userQuery }}</b></div>

      <div class="sec"><span class="n">进度</span><div class="ln"></div></div>
      <div class="timeline">
        <div v-for="s in steps" :key="s.key" class="node" :class="s.status">
          <h4>
            {{ s.title }}
            <span v-if="s.status === 'done'" class="badge done">完成</span>
            <span v-else-if="s.status === 'run'" class="badge run">进行中</span>
            <span v-else-if="s.status === 'error'" class="badge err">出错</span>
          </h4>
          <p v-if="s.detail">{{ s.detail }}</p>
        </div>
      </div>

      <template v-if="reviewMarkdown">
        <div class="sec"><span class="n">综述</span><div class="ln"></div></div>
        <div v-if="reviewWarning" class="err-inline soft-warning">{{ reviewWarning }}</div>
        <article class="paper">
          <div class="review-body" v-html="renderedReview()"></div>
        </article>

        <!-- 忠实度报告：仍不达标的论断要让用户感知 -->
        <div v-if="faithReport" class="faith-report" :class="faithReport.verdict">
          <div class="fr-head">
            <span class="fr-icon">{{ faithReport.verdict === 'pass' ? '✓' : '!' }}</span>
            忠实度校验：{{ faithReport.verdict === 'pass' ? '全部通过' : '带警告通过' }}
            <span class="fr-stat">共 {{ faithReport.total_claims }} 条论断 · 修订 {{ faithReport.revised }} 条 · 仍不达标 {{ faithReport.remaining_bad }} 条</span>
          </div>
          <div v-if="faithReport.detail && faithReport.detail.length" class="fr-detail">
            <div class="fr-detail-title">以下论断证据支撑不足，已尽力保守表述，请人工复核：</div>
            <div v-for="(d, i) in faithReport.detail" :key="i" class="fr-item">{{ d }}</div>
          </div>
        </div>

        <div class="actions center">
          <button class="ghost" @click="reset">重新生成</button>
          <button class="primary" @click="downloadMd">下载 Markdown</button>
        </div>
      </template>
    </section>

    <!-- 错误态 -->
    <section v-else-if="phase === 'error'" class="card err-card">
      <h2 class="card-title">生成中断</h2>
      <p class="card-hint">{{ errorMsg }}</p>
      <div class="actions"><button class="primary" @click="reset">返回重试</button></div>
    </section>
  </div>
</template>

<style>
@import '../assets/review.css';
@import '../assets/proposal.css';
</style>
