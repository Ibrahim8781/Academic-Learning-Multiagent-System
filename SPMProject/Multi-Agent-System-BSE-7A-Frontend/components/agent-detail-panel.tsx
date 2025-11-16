"use client"

import { useSettings } from "@/context/settings-context"
import { X, Copy, CheckCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { useState } from "react"
import { Agent } from "@/types/agent"

interface AgentDetailPanelProps {
  agent: Agent
  onClose?: () => void
}

export default function AgentDetailPanel({ agent, onClose }: AgentDetailPanelProps) {
  const { settings, updateSettings } = useSettings()
  const [copied, setCopied] = useState(false)

  const handleCopyId = () => {
    navigator.clipboard.writeText(agent.id)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-border flex items-center justify-between gap-2">
        <h3 className="font-semibold text-sm">Agent Details</h3>
        {onClose && (
          <Button variant="ghost" size="icon" onClick={onClose} className="focus-ring" aria-label="Close detail panel">
            <X className="w-4 h-4" />
          </Button>
        )}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Agent Info */}
        <div className="space-y-2">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase">Agent Information</h4>
          <div className="space-y-2">
            <div>
              <p className="text-xs text-muted-foreground">Name</p>
              <p className="text-sm font-medium">{agent.name}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Description</p>
              <p className="text-sm">{agent.description}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">ID</p>
              <div className="flex items-center gap-2">
                <code className="text-xs bg-muted px-2 py-1 rounded flex-1 truncate">{agent.id}</code>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={handleCopyId}
                  className="h-6 w-6 focus-ring"
                  aria-label="Copy agent ID"
                >
                  {copied ? <CheckCircle className="w-3 h-3 text-green-600" /> : <Copy className="w-3 h-3" />}
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Capabilities */}
        {agent.capabilities && agent.capabilities.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-xs font-semibold text-muted-foreground uppercase">Capabilities</h4>
            <div className="flex flex-wrap gap-1">
              {agent.capabilities.map((cap) => (
                <span key={cap} className="text-xs px-2 py-1 rounded-full bg-accent/20 text-accent">
                  {cap}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Settings */}
        <div className="space-y-3 border-t border-border pt-4">
          <h4 className="text-xs font-semibold text-muted-foreground uppercase">Execution Settings</h4>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={settings.enableLTM}
              onChange={(e) => updateSettings({ ...settings, enableLTM: e.target.checked })}
              className="rounded focus-ring"
              aria-label="Use long-term memory"
            />
            <span className="text-sm">Use Long-Term Memory (LTM)</span>
          </label>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={settings.enableSTM}
              onChange={(e) => updateSettings({ ...settings, enableSTM: e.target.checked })}
              className="rounded focus-ring"
              aria-label="Use short-term memory"
            />
            <span className="text-sm">Use Short-Term Memory (STM)</span>
          </label>

          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={settings.autoRoute}
              onChange={(e) => updateSettings({ ...settings, autoRoute: e.target.checked })}
              className="rounded focus-ring"
              aria-label="Enable auto-routing"
            />
            <span className="text-sm">Enable Auto-Routing</span>
          </label>
        </div>
      </div>
    </div>
  )
}
