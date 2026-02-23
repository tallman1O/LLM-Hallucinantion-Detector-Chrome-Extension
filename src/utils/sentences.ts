/**
 * Split text into sentences and return the first and last.
 */
export function getFirstAndLastSentence(text: string): {
  firstSentence: string
  lastSentence: string
} {
  const trimmed = text.trim()
  if (!trimmed) return { firstSentence: '', lastSentence: '' }
  const sentences = trimmed.split(/(?<=[.!?])\s+/).filter(Boolean)
  if (sentences.length === 0) return { firstSentence: trimmed, lastSentence: trimmed }
  const first = sentences[0]!.trim()
  const last = sentences[sentences.length - 1]!.trim()
  return { firstSentence: first, lastSentence: last }
}

/**
 * Phase 1: dummy text that "matches" first and last sentence.
 */
export function getDummyMatchText(firstSentence: string, lastSentence: string): string {
  if (!firstSentence && !lastSentence) return ''
  if (!lastSentence) return firstSentence
  if (!firstSentence) return lastSentence
  if (firstSentence === lastSentence) return firstSentence
  return `${firstSentence} … ${lastSentence}`
}
