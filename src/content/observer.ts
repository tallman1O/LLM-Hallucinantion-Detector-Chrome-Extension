/**
 * ChatGPT DOM: assistant message text from last assistant turn.
 * Selectors may need updating if ChatGPT changes markup.
 */
export function getLastAssistantMessageText(): string {
  const selectors = [
    '[data-message-author-role="assistant"]',
    'main [class*="markdown"]',
    'main [class*="prose"]',
    'main article',
    '[data-testid="conversation-turn"]',
  ]
  for (const sel of selectors) {
    const nodes = document.querySelectorAll(sel)
    const last = nodes[nodes.length - 1]
    if (last) {
      const text = (last as HTMLElement).innerText?.trim() ?? (last as HTMLElement).textContent?.trim() ?? ''
      if (text.length > 0) return text
    }
  }
  return ''
}

export function debounce<T extends (...args: unknown[]) => void>(fn: T, ms: number): (...args: Parameters<T>) => void {
  let id: ReturnType<typeof setTimeout>
  return (...args: Parameters<T>) => {
    clearTimeout(id)
    id = setTimeout(() => fn(...args), ms)
  }
}
