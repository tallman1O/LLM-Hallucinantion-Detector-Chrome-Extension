function App() {
  return (
    <div className="flex min-h-screen flex-col gap-6 bg-[#0f0f12] px-5 py-6 text-[#e4e4e7] dark:bg-[#0f0f12]">
      <header className="border-b border-[#2a2a2e] pb-4">
        <h1 className="mb-1 text-xl font-semibold tracking-tight text-[#e4e4e7]">
          LLM Hallucination Detector
        </h1>
        <p className="text-[13px] font-medium text-[#a78bfa]">
          Spot potential hallucinations in AI replies
        </p>
      </header>
      <section className="flex flex-1 flex-col gap-4">
        <p className="text-sm leading-relaxed text-[#71717a]">
          This extension detects potential hallucinations in responses from LLM models like ChatGPT, Claude, and Gemini. Open a conversation to analyze responses.
        </p>
        <div className="mt-2 rounded-xl border border-[#2a2a2e] bg-[#18181c] p-4">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-[#71717a]">
            Last analysis
          </p>
          <p className="text-[13px] text-[#71717a]">
            Analyze a response on ChatGPT to see results here.
          </p>
        </div>
      </section>
    </div>
  )
}

export default App
