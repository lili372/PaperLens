<script setup>
import { computed, reactive, ref } from 'vue'
import { marked } from 'marked'
import DOMPurify from 'dompurify'

defineProps({
  activeMode: { type: String, default: 'proposal' },
})
const emit = defineEmits(['switch-mode'])

const phase = ref('input')
const inputMode = ref('normal')
const userRequest = ref('偏多标记学习领域的技术探索报告')
const errorMsg = ref('')
const sessionId = ref('')
let eventSource = null

const plan = reactive({
  base_query: 'partial multi-label learning',
  module_terms: ['label correlation', 'pseudo label', 'label noise'],
  start_date: '2023-01-01',
  end_date: '',
  rationale: '',
})

const paperReview = reactive({
  raw_count: 0,
  target_success_count: 0,
  pdf_success_count: 0,
  pdf_failed_count: 0,
  pdf_skipped_count: 0,
  not_selected_count: 0,
  papers: [],
})

const steps = ref([])
const STEP_DEFS = [
  { key: 'search', title: '检索 arXiv' },
  { key: 'pdf_download', title: '下载 PDF' },
  { key: 'profiles', title: '生成论文画像' },
  { key: 'topic', title: '生成候选方向' },
  { key: 'review_batch', title: '查证与可做性审查' },
  { key: 'revision_batch', title: '单次修改与分流' },
  { key: 'report', title: '生成最终报告' },
]

const report = ref(null)
const reportMarkdown = ref('')
const selectedCandidateId = ref('')
const detachedPlanReview = ref(false)

const finalCandidates = computed(() => report.value?.primary_candidates || report.value?.final_candidates || [])
const secondaryCandidates = computed(() => report.value?.secondary_candidates || [])
const deferredCandidates = computed(() => report.value?.deferred_candidates || [])
const selectedCandidate = computed(() => {
  return finalCandidates.value.find(item => item.candidate_id === selectedCandidateId.value) || finalCandidates.value[0] || null
})

function newSessionId() {
  return `proposal_${Date.now()}_${Math.random().toString(16).slice(2)}`
}

function normalizedPlan() {
  return {
    base_query: plan.base_query.trim(),
    module_terms: plan.module_terms.map(s => s.trim()).filter(Boolean).slice(0, 3),
    start_date: plan.start_date || null,
    end_date: plan.end_date || null,
    rationale: plan.rationale || null,
  }
}

function start() {
  errorMsg.value = ''
  report.value = null
  reportMarkdown.value = ''
  selectedCandidateId.value = ''
  detachedPlanReview.value = false
  steps.value = []
  paperReview.papers = []
  sessionId.value = newSessionId()

  const params = new URLSearchParams()
  params.set('session_id', sessionId.value)
  params.set('mode', inputMode.value)
  params.set('user_request', userRequest.value.trim())
  if (inputMode.value === 'expert') {
    params.set('plan', JSON.stringify(normalizedPlan()))
  }

  phase.value = inputMode.value === 'expert' ? 'searching_downloading' : 'connecting'
  eventSource = new EventSource(`/api/proposal?${params.toString()}`)
  eventSource.onmessage = (e) => {
    try { handle(JSON.parse(e.data)) } catch (err) { console.error(err) }
  }
  eventSource.onerror = () => fail('与服务器的连接中断')
}

function handle({ step, state, data }) {
  if (state !== 'generating') console.log(`[V2 SSE] step=${step} state=${state}`, data)
  switch (state) {
    case 'plan_review': return onPlanReview(data)
    case 'paper_review': return onPaperReview(data)
    case 'initializing':
    case 'thinking': return onProgress(step, data)
    case 'completed': return onStepCompleted(step, data)
    case 'result': return onResult(data)
    case 'error': return fail(data)
    case 'finished': return closeSSE()
  }
}

function onPlanReview(raw) {
  const obj = typeof raw === 'string' ? JSON.parse(raw) : raw
  plan.base_query = obj.base_query || ''
  plan.module_terms = [...(obj.module_terms || []), '', '', ''].slice(0, 3)
  plan.start_date = obj.start_date || ''
  plan.end_date = obj.end_date || ''
  plan.rationale = obj.rationale || ''
  phase.value = 'plan_review'
}

function confirmPlan() {
  if (detachedPlanReview.value || !eventSource) {
    inputMode.value = 'expert'
    start()
    return
  }
  sendProposalInput({ type: 'plan', plan: normalizedPlan() })
  phase.value = 'searching_downloading'
  initSteps()
}

function onPaperReview(raw) {
  const obj = typeof raw === 'string' ? JSON.parse(raw) : raw
  paperReview.raw_count = obj.raw_count || 0
  paperReview.target_success_count = obj.target_success_count || obj.pdf_success_count || 0
  paperReview.pdf_success_count = obj.pdf_success_count || 0
  paperReview.pdf_failed_count = obj.pdf_failed_count || 0
  paperReview.pdf_skipped_count = obj.pdf_skipped_count || 0
  paperReview.not_selected_count = obj.not_selected_count || 0
  paperReview.papers = obj.papers || []
  phase.value = 'papers_review'
}

function confirmPapers() {
  sendProposalInput({ type: 'papers', decision: 'continue' })
  phase.value = 'running'
  initSteps()
}

function retryFailedPdfs() {
  sendProposalInput({ type: 'papers', decision: 'retry_failed_pdfs' })
  phase.value = 'searching_downloading'
  initSteps()
  const item = findStep('pdf_download')
  if (item) {
    item.status = 'run'
    item.detail = '正在重新下载失败 PDF'
  }
}

function abortPapers() {
  sendProposalInput({ type: 'papers', decision: 'abort' })
  closeSSE()
  detachedPlanReview.value = true
  phase.value = 'plan_review'
}

function initSteps() {
  if (!steps.value.length) {
    steps.value = STEP_DEFS.map(item => ({ ...item, status: 'wait', detail: '' }))
  }
}

function onProgress(step, data) {
  if (!isVisibleStep(step)) return
  initSteps()
  if (phase.value === 'connecting') phase.value = 'searching_downloading'
  const item = findStep(step)
  if (item) {
    item.status = 'run'
    item.detail = typeof data === 'string' ? data : ''
  }
}

function onStepCompleted(step, data) {
  if (!isVisibleStep(step)) return
  initSteps()
  const item = findStep(step)
  if (item) {
    item.status = 'done'
    item.detail = typeof data === 'string' ? data : ''
  }
}

function onResult(raw) {
  const obj = typeof raw === 'string' ? JSON.parse(raw) : raw
  report.value = obj.report
  reportMarkdown.value = obj.markdown || ''
  selectedCandidateId.value = report.value?.final_candidates?.[0]?.candidate_id || ''
  phase.value = 'result'
}

async function loadLatestReport() {
  errorMsg.value = ''
  closeSSE()
  try {
    const response = await fetch('/api/proposal/latest_report')
    if (!response.ok) throw new Error('没有找到可展示的最新报告')
    const obj = await response.json()
    report.value = obj.report
    reportMarkdown.value = obj.markdown || ''
    userRequest.value = report.value?.scope?.user_request || userRequest.value
    const candidates = report.value?.primary_candidates || report.value?.final_candidates || []
    selectedCandidateId.value = candidates[0]?.candidate_id || ''
    steps.value = STEP_DEFS.map(item => ({ ...item, status: 'done', detail: '' }))
    phase.value = 'result'
  } catch (err) {
    fail(err.message || '加载最新报告失败')
  }
}

function findStep(step) {
  return steps.value.find(item => item.key === step)
}

function isVisibleStep(step) {
  return STEP_DEFS.some(item => item.key === step)
}

async function sendProposalInput(payload) {
  await fetch('/proposal_input', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId.value, ...payload }),
  })
}

function fail(msg) {
  errorMsg.value = typeof msg === 'string' ? msg : '生成失败'
  const running = steps.value.find(item => item.status === 'run')
  if (running) running.status = 'error'
  if (!['plan_review', 'papers_review'].includes(phase.value)) phase.value = 'error'
  closeSSE()
}

function closeSSE() {
  if (eventSource) {
    eventSource.close()
    eventSource = null
  }
}

function reset() {
  closeSSE()
  phase.value = 'input'
  errorMsg.value = ''
  report.value = null
  reportMarkdown.value = ''
  selectedCandidateId.value = ''
  detachedPlanReview.value = false
  steps.value = []
}

function addModuleTerm() {
  if (plan.module_terms.length < 3) plan.module_terms.push('')
}

function removeModuleTerm(index) {
  plan.module_terms.splice(index, 1)
  if (!plan.module_terms.length) plan.module_terms.push('')
}

function downloadMd() {
  const content = reportMarkdown.value || '# V2 研究方向报告\n'
  const baseQuery = report.value?.scope?.base_query || 'PaperLens-V2'
  const safeBaseQuery = baseQuery.trim().replace(/[\\/:*?"<>|]+/g, '_').replace(/\s+/g, '_') || 'PaperLens-V2'
  const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `${safeBaseQuery}_研究方向报告.md`
  a.click()
  URL.revokeObjectURL(url)
}

function renderedMarkdown() {
  return DOMPurify.sanitize(marked.parse(reportMarkdown.value || ''))
}
</script>

<template>
  <div class="page proposal-page">
    <header class="top">
      <div class="crest">P</div>
      <div class="name">Paper<span>Lens</span></div>
      <nav class="mode-tabs" aria-label="功能模式">
        <button type="button" @click="emit('switch-mode', 'review')">综述分析</button>
        <button class="active" type="button">研究方向推荐</button>
      </nav>
      <div class="tag">候选选题探索</div>
    </header>

    <section v-if="phase === 'input'" class="hero proposal-hero">
      <h1>从近期论文里，找到<span class="u">值得精读的研究方向</span></h1>
      <p class="lead">输入研究需求，系统检索 arXiv、下载 PDF、生成论文画像，并查证候选方向是否值得继续做小实验。</p>

      <div class="switch-line">
        <button :class="{ active: inputMode === 'normal' }" type="button" @click="inputMode = 'normal'">普通模式</button>
        <button :class="{ active: inputMode === 'expert' }" type="button" @click="inputMode = 'expert'">专家模式</button>
      </div>

      <div v-if="inputMode === 'normal'" class="proposal-input">
        <textarea v-model="userRequest" placeholder="例如：偏多标记学习领域的技术探索报告"></textarea>
        <button @click="start">生成检索计划</button>
      </div>

      <div v-else class="expert-box">
        <label>
          研究需求
          <input v-model="userRequest" placeholder="偏多标记学习领域的技术探索报告" />
        </label>
        <label>
          领域本体词
          <input v-model="plan.base_query" placeholder="partial multi-label learning" />
        </label>
        <div class="module-list">
          <label v-for="(_, i) in plan.module_terms" :key="i">
            模块词 {{ i + 1 }}
            <span>
              <input v-model="plan.module_terms[i]" placeholder="label correlation" />
              <button type="button" @click="removeModuleTerm(i)">×</button>
            </span>
          </label>
          <button v-if="plan.module_terms.length < 3" class="kw-add" type="button" @click="addModuleTerm">+ 添加模块词</button>
        </div>
        <div class="date-grid">
          <label>开始时间<input v-model="plan.start_date" placeholder="2023-01-01" /></label>
          <label>结束时间<input v-model="plan.end_date" placeholder="可留空" /></label>
        </div>
        <div class="actions center"><button class="primary" @click="start">直接搜索并下载 PDF</button></div>
      </div>
      <div class="actions center latest-actions">
        <button class="ghost" type="button" @click="loadLatestReport">查看最新报告</button>
      </div>
    </section>

    <section v-else-if="phase === 'connecting' || phase === 'searching_downloading'" class="wait-box">
      <div class="orig-q">研究需求：<b>{{ userRequest }}</b></div>
      <div class="spinner"></div>
      <p>{{ phase === 'connecting' ? '正在生成检索计划…' : '正在搜索 arXiv 并下载 PDF…' }}</p>
      <div v-if="steps.length" class="timeline mini">
        <div v-for="s in steps" :key="s.key" class="node" :class="s.status">
          <h4>{{ s.title }}<span v-if="s.status === 'done'" class="badge done">完成</span><span v-else-if="s.status === 'run'" class="badge run">进行中</span></h4>
          <p v-if="s.detail">{{ s.detail }}</p>
        </div>
      </div>
    </section>

    <section v-else-if="phase === 'plan_review'" class="card">
      <div class="orig-q">研究需求：<b>{{ userRequest }}</b></div>
      <h2 class="card-title">确认 V2 检索计划</h2>
      <p class="card-hint">领域本体词会单独检索，模块词会与领域本体词组合检索。最多保留 3 个模块词。</p>
      <label class="form-row">领域本体词<input v-model="plan.base_query" /></label>
      <div class="kw-list vertical">
        <label v-for="(_, i) in plan.module_terms" :key="i" class="form-row">
          模块词 {{ i + 1 }}
          <span class="inline-control"><input v-model="plan.module_terms[i]" /><button class="kw-del" @click="removeModuleTerm(i)">×</button></span>
        </label>
        <button v-if="plan.module_terms.length < 3" class="kw-add" @click="addModuleTerm">+ 添加模块词</button>
      </div>
      <div class="date-grid">
        <label class="form-row">开始时间<input v-model="plan.start_date" /></label>
        <label class="form-row">结束时间<input v-model="plan.end_date" placeholder="不限" /></label>
      </div>
      <p v-if="plan.rationale" class="rationale">{{ plan.rationale }}</p>
      <div class="actions">
        <button class="ghost" @click="reset">返回修改需求</button>
        <button class="primary" @click="confirmPlan">确认，开始搜索</button>
      </div>
    </section>

    <section v-else-if="phase === 'papers_review'" class="card wide-card">
      <div class="orig-q">研究需求：<b>{{ userRequest }}</b></div>
      <h2 class="card-title">确认进入分析的论文</h2>
      <p class="card-hint">
        本次 arXiv 原始去重召回 <b>{{ paperReview.raw_count }}</b> 篇，
        目标进入分析 <b>{{ paperReview.target_success_count }}</b> 篇，
        实际进入分析 <b>{{ paperReview.pdf_success_count }}</b> 篇；
        PDF 下载失败 <b>{{ paperReview.pdf_failed_count }}</b> 篇，
        因篇幅或综述标题跳过 <b>{{ paperReview.pdf_skipped_count }}</b> 篇，
        未进入分析 <b>{{ paperReview.not_selected_count }}</b> 篇。请确认整体是否对路。
      </p>
      <ul class="paper-list proposal-paper-list">
        <li v-for="(p, i) in paperReview.papers" :key="p.paper_id || i">
          <span class="p-idx">{{ i + 1 }}</span>
          <span class="p-main">
            <span class="p-title">{{ p.title }}</span>
            <span class="p-meta">{{ p.paper_id }} · {{ p.published }} · {{ (p.matched_queries || []).join(' / ') }}</span>
          </span>
          <span class="p-year">{{ p.primary_category || '' }}</span>
        </li>
      </ul>
      <div class="actions">
        <button class="ghost" @click="abortPapers">返回修改检索计划</button>
        <button v-if="paperReview.pdf_failed_count > 0" class="ghost" @click="retryFailedPdfs">重新下载失败PDF</button>
        <button class="primary" @click="confirmPapers">看着对，继续分析</button>
      </div>
    </section>

    <section v-else-if="phase === 'running' || phase === 'result'">
      <div class="orig-q big">研究需求：<b>{{ userRequest }}</b></div>
      <div class="sec"><span class="n">V2 进度</span><div class="ln"></div></div>
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

      <template v-if="report">
        <div class="sec"><span class="n">推荐方向</span><div class="ln"></div></div>
        <div class="proposal-summary">
          <h2>{{ report.scope?.user_request || '候选研究方向报告' }}</h2>
          <p>
            {{ report.scope?.search_paper_count }} 篇召回 ·
            {{ report.scope?.pdf_success_count }} 篇 PDF 分析 ·
            {{ report.scope?.final_candidate_count }} 个推荐方向 ·
            {{ report.scope?.deferred_candidate_count }} 个暂缓方向
          </p>
        </div>

        <div class="candidate-tabs">
            <button
              v-for="item in finalCandidates"
              :key="item.candidate_id"
              :class="{ active: selectedCandidate?.candidate_id === item.candidate_id }"
              type="button"
              @click="selectedCandidateId = item.candidate_id"
            >
              <span>{{ item.candidate_id }}</span>
              <b>{{ item.candidate_title || item.candidate_id }}</b>
              <small>建议先读：{{ (item.target_paper_ids || []).join('、') || '暂无' }}</small>
            </button>
        </div>

        <div v-if="!finalCandidates.length" class="deferred-box">
          <h3>当前材料没有形成可靠推荐方向</h3>
          <p>{{ report.empty_recommendation_message || '建议扩大检索范围、调整模块词，或补充更多可分析论文后重试。' }}</p>
        </div>

        <article v-if="selectedCandidate" class="candidate-detail">
            <h2>{{ selectedCandidate.candidate_id }}：{{ selectedCandidate.candidate_title || '候选研究方向' }}</h2>
            <div class="read-first">建议先读：{{ (selectedCandidate.target_paper_ids || []).join('、') || '暂无' }}</div>
            <section>
              <h3>方向说明</h3>
              <p>{{ selectedCandidate.candidate_direction }}</p>
            </section>
            <section>
              <h3>为什么值得看</h3>
              <p>{{ selectedCandidate.why_worth_reading }}</p>
            </section>
            <section>
              <h3>研究价值</h3>
              <p>{{ selectedCandidate.research_value }}</p>
            </section>
            <section>
              <h3>证据支持</h3>
              <ul>
                <li v-for="(claim, key) in selectedCandidate.claim_checks" :key="key">
                  <b>{{ key }}</b>：{{ claim.status }}。{{ claim.note }}
                </li>
              </ul>
            </section>
            <section>
              <h3>关键支持材料</h3>
              <ul>
                <li v-for="(e, i) in selectedCandidate.supporting_evidence" :key="i">
                  {{ e.supported_claim }}：{{ e.evidence_summary }}
                </li>
              </ul>
            </section>
            <section>
              <h3>主要风险</h3>
              <ul>
                <li v-for="(e, i) in selectedCandidate.risk_evidence" :key="i">
                  {{ e.risk_point }}：{{ e.evidence_summary }}
                </li>
              </ul>
            </section>
            <section>
              <h3>建议先做的验证</h3>
              <p>{{ selectedCandidate.validation_path || selectedCandidate.final_advice }}</p>
            </section>
        </article>

        <div v-if="deferredCandidates.length" class="deferred-box">
          <h3>未进入主推荐的方向</h3>
          <p v-for="item in deferredCandidates" :key="item.candidate_id">
            <b>{{ item.candidate_id }}：{{ item.candidate_title || item.candidate_direction }}</b>。
            方向说明：{{ item.candidate_direction }}。暂缓原因：{{ item.defer_reason }}
          </p>
        </div>

        <div v-if="secondaryCandidates.length" class="deferred-box">
          <h3>其他可考虑方向</h3>
          <p v-for="item in secondaryCandidates" :key="item.candidate_id">
            <b>{{ item.candidate_id }}：{{ item.candidate_title || '候选研究方向' }}</b>。
            {{ item.candidate_direction }}
          </p>
        </div>

        <details class="raw-report">
          <summary>查看 Markdown 原报告</summary>
          <article class="paper"><div class="review-body" v-html="renderedMarkdown()"></div></article>
        </details>

        <div class="actions center">
          <button class="ghost" @click="reset">重新生成</button>
          <button class="primary" @click="downloadMd">下载 Markdown</button>
        </div>
      </template>
    </section>

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
