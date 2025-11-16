"use client"

import { useHistory } from "@/context/history-context"
import { Trash2, Copy, RotateCcw, Download, X, Search } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useState, useMemo } from "react"

interface HistoryPanelProps {
  agentId: string
  onClose?: () => void
  onClear?: () => void
}

export default function HistoryPanel({ agentId, onClose, onClear }: HistoryPanelProps) {
  const { getHistory, clearHistory } = useHistory()
  const [searchTerm, setSearchTerm] = useState("")
  const messages = getHistory(agentId)

  const handleClearHistory = () => {
    clearHistory(agentId)
    if (onClear) {
      onClear()
    }
  }

  const filteredMessages = useMemo(() => {
    if (!messages) return []
    return messages.filter((msg) => {
      const content = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content)
      return content.toLowerCase().includes(searchTerm.toLowerCase())
    })
  }, [messages, searchTerm])

  const handleCopyRequest = (content: any) => {
    const textToCopy = typeof content === "string" ? content : JSON.stringify(content, null, 2)
    navigator.clipboard.writeText(textToCopy)
  }

  const handleExportResult = (content: any) => {
    const textToExport = typeof content === "string" ? content : JSON.stringify(content, null, 2)
    const element = document.createElement("a")
    element.setAttribute("href", "data:text/plain;charset=utf-8," + encodeURIComponent(textToExport))
    element.setAttribute("download", `agent-result-${Date.now()}.txt`)
    element.style.display = "none"
    document.body.appendChild(element)
    element.click()
    document.body.removeChild(element)
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-border flex items-center justify-between gap-2">
        <h3 className="font-semibold text-sm">History</h3>
        {onClose && (
          <Button variant="ghost" size="icon" onClick={onClose} className="focus-ring" aria-label="Close history panel">
            <X className="w-4 h-4" />
          </Button>
        )}
      </div>

      {/* Search */}
      <div className="p-3 border-b border-border">
        <div className="relative">
          <Search className="absolute left-2 top-2.5 w-4 h-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search history..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full pl-8 pr-3 py-2 text-sm rounded-md border border-border bg-background focus:outline-none focus:ring-2 focus:ring-primary"
            aria-label="Search history"
          />
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {filteredMessages.length === 0 ? (
          <p className="text-xs text-muted-foreground text-center py-4">
            {!messages || messages.length === 0 ? "No history yet" : "No matching results"}
          </p>
        ) : (
          filteredMessages.map((msg, idx) => (
            <div key={idx} className="text-xs p-2 rounded-md bg-muted/50 border border-border/50 space-y-1">
              <div className="flex items-center justify-between gap-1">
                <p className="font-semibold text-primary capitalize">{msg.type}</p>
                <div className="flex gap-1">
                  {msg.type === "user" && (
                    <>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => handleCopyRequest(msg.content)}
                        className="h-6 w-6"
                        title="Copy request"
                        aria-label="Copy request"
                      >
                        <Copy className="w-3 h-3" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-6 w-6"
                        title="Re-run request"
                        aria-label="Re-run request"
                      >
                        <RotateCcw className="w-3 h-3" />
                      </Button>
                    </>
                  )}
                  {msg.type === "agent" && (
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleExportResult(msg.content)}
                      className="h-6 w-6"
                      title="Export result"
                      aria-label="Export result"
                    >
                      <Download className="w-3 h-3" />
                    </Button>
                  )}
                </div>
              </div>
              <p className="text-muted-foreground line-clamp-2">
                {typeof msg.content === "string" ? msg.content : msg.content.summary || JSON.stringify(msg.content)}
              </p>
              <p className="text-xs text-muted-foreground/60">{new Date(msg.timestamp).toLocaleTimeString()}</p>
            </div>
          ))
        )}
      </div>

      {/* Footer */}
      {messages && messages.length > 0 && (
        <div className="p-3 border-t border-border">
          <Button
            variant="destructive"
            size="sm"
            onClick={handleClearHistory}
            className="w-full text-xs focus-ring"
            aria-label="Clear all history"
          >
            <Trash2 className="w-3 h-3 mr-1" />
            Clear History
          </Button>
        </div>
      )}
    </div>
  )
}
