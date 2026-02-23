import { useState, useEffect, useCallback } from 'react'
import { getFirstAndLastSentence, getDummyMatchText } from '../utils/sentences'
import { getLastAssistantMessageText, debounce } from './observer'
import './content.css'

const DEBOUNCE_MS = 400
const STABLE_MS = 800

export default function ContentPage() {
  const [responseText, setResponseText] = useState('')
  const [isPanelOpen, setIsPanelOpen] = useState(false)

  const onToggle = useCallback(() => setIsPanelOpen((o) => !o), [])

  useEffect(() => {
    let stableTimer: ReturnType<typeof setTimeout> | null = null

    const checkText = () => {
      const text = getLastAssistantMessageText()
      clearTimeout(stableTimer!)
      if (text) stableTimer = setTimeout(() => setResponseText(text), STABLE_MS)
    }

    const debouncedCheck = debounce(checkText, DEBOUNCE_MS)
    const observer = new MutationObserver(() => debouncedCheck())

    observer.observe(document.body, {
      childList: true,
      subtree: true,
      characterData: true,
      characterDataOldValue: true,
    })
    checkText()

    return () => {
      observer.disconnect()
      if (stableTimer) clearTimeout(stableTimer)
    }
  }, [])

  const hasResponse = responseText.length > 0
  const { firstSentence, lastSentence } = getFirstAndLastSentence(responseText)
  const dummyMatch = getDummyMatchText(firstSentence, lastSentence)

  return (
    <>
      {hasResponse && (
        <button
          type="button"
          className={`fixed ${!isPanelOpen ? "animate-pulse" : ""} bottom-6 right-6 z-2147483645 flex h-[52px] w-[52px] items-center justify-center rounded-full border border-[#2a2a2e] bg-[#18181c] text-2xl font-light text-[#e4e4e7] shadow-lg transition-all hover:scale-105 hover:border-[#a78bfa] hover:bg-[#2a2a2e] focus:border-[#a78bfa] focus:outline-none ${isPanelOpen ? 'bg-[#a78bfa]' : ''}`}
          onClick={onToggle}
          aria-expanded={isPanelOpen}
          aria-label={isPanelOpen ? 'Close analysis panel' : 'Open analysis panel'}
        >
          {isPanelOpen ? '👀' : '🧠'}
        </button>
      )}
      {hasResponse && isPanelOpen && (
        <div
          className="fixed right-6 z-2147483646 max-h-[70vh] w-[360px] max-w-[calc(100vw-48px)] overflow-auto rounded-xl border border-[#2a2a2e] bg-[#18181c] p-4 pb-5 text-sm leading-normal text-[#e4e4e7] shadow-2xl"
          role="dialog"
          aria-label="Response analysis"
          style={{ bottom: '88px' }}
        >
          <h2 className="mb-4 text-xs font-semibold uppercase tracking-wider text-[#a78bfa]">
            Response analysis
          </h2>
          <div className="mb-4 last:mb-0">
            <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-[#71717a]">
              First sentence
            </p>
            <p className="wrap-break-word text-[#e4e4e7]">
              {firstSentence || <span className="italic text-[#71717a]">—</span>}
            </p>
          </div>
          <div className="mb-4 last:mb-0">
            <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-[#71717a]">
              Last sentence
            </p>
            <p className="wrap-break-word text-[#e4e4e7]">
              {lastSentence || <span className="italic text-[#71717a]">—</span>}
            </p>
          </div>
          <div className="mb-4 last:mb-0">
            <p className="mb-1.5 text-[11px] font-semibold uppercase tracking-wider text-[#71717a]">
              Dummy match
            </p>
            <p className="wrap-break-word text-[#e4e4e7]">
              {dummyMatch || <span className="italic text-[#71717a]">—</span>}
            </p>
          </div>
        </div>
      )}
    </>
  )
}
