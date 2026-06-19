<script setup>
import { ref } from 'vue'
import ReviewPage from './pages/ReviewPage.vue'
import ProposalPage from './pages/ProposalPage.vue'

const activeMode = ref('review')
</script>

<template>
  <ReviewPage v-if="activeMode === 'review'" :active-mode="activeMode" @switch-mode="activeMode = $event" />
  <ProposalPage v-else :active-mode="activeMode" @switch-mode="activeMode = $event" />
</template>
