import { createRoot } from 'react-dom/client'
import ContentPage from './content/content-page'

/**
 * Injects the extension UI into the page without modifying any existing HTML.
 * Creates a container div and mounts ContentPage (circular toggle + panel).
 */
function bootstrap() {
  const host = document.createElement('div')
  host.id = 'llm-hallucination-detector-root'
  host.style.cssText = 'position:fixed;inset:0;pointer-events:none;z-index:2147483644;'
  const inner = document.createElement('div')
  inner.style.cssText = 'position:fixed;bottom:0;right:0;pointer-events:auto;'
  host.appendChild(inner)
  document.body.appendChild(host)

  const root = createRoot(inner)
  root.render(<ContentPage />)
}

bootstrap()
