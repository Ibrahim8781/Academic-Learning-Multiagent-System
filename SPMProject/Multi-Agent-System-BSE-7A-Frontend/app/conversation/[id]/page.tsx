"use client"

import { useParams } from "next/navigation"
import Link from "next/link"
import { useState } from "react"
import Header from "@/components/header"
import { useAgents } from "@/context/agent-context"
import ChatWindow from "@/components/chat-window"
import { OfflineWarning } from "@/components/offline-warning"
import { Button } from "@/components/ui/button"
import AgentDetailPanel from "@/components/agent-detail-panel"
import HistoryPanel from "@/components/history-panel"
import { Sheet, SheetContent } from "@/components/ui/sheet"
import { useIsMobile } from "@/hooks/use-mobile"
import { submitSupervisorRequest } from "@/lib/api-service"
import { useHistory } from "@/context/history-context"
import { type RequestPayload } from "@/types"

export default function ConversationPage() {
  const params = useParams()
  const agentId = params.id as string
  const { agents, agentHealth } = useAgents()
  const [selectedAlternative, setSelectedAlternative] = useState<string | null>(null)
  const [showHistory, setShowHistory] = useState(false)
  const [showDetailPanel, setShowDetailPanel] = useState(false)
  const isMobile = useIsMobile()
  const { addMessage } = useHistory()
  const [loading, setLoading] = useState(false)

  const agent = agents.find((a) => a.id === agentId)
  const health = agentHealth[agentId]
  const isOffline = health === "offline"

  if (!agent) {
    return (
      <div className="flex h-screen bg-background text-foreground">
        <div className="flex-1 flex flex-col">
          <Header />
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <h2 className="text-2xl font-bold mb-2">Agent Not Found</h2>
              <Link href="/dashboard">
                <Button>Back to Dashboard</Button>
              </Link>
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (isOffline && !selectedAlternative) {
    const alternativeAgents = agents.filter((a) => a.id !== agentId && agentHealth[a.id] !== "offline")

    return (
      <div className="flex h-screen bg-background text-foreground">
        <div className="flex-1 flex flex-col">
          <Header />
          <div className="flex-1 flex items-center justify-center p-6">
            <div className="max-w-md space-y-4">
              <OfflineWarning
                agentName={agent.name}
                onSelectAlternative={() => {
                  if (alternativeAgents.length > 0) {
                    setSelectedAlternative(alternativeAgents[0].id)
                  }
                }}
              />
              {alternativeAgents.length > 0 && (
                <div className="space-y-2">
                  <p className="text-sm font-medium">Available alternatives:</p>
                  {alternativeAgents.map((alt) => (
                    <Button
                      key={alt.id}
                      variant="outline"
                      className="w-full justify-start bg-transparent"
                      onClick={() => setSelectedAlternative(alt.id)}
                    >
                      {alt.name}
                    </Button>
                  ))}
                </div>
              )}
              <Link href="/agents" className="block">
                <Button variant="ghost" className="w-full">
                  Browse All Agents
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </div>
    )
  }

  const displayAgentId = selectedAlternative || agentId
  const displayAgent = agents.find((a) => a.id === displayAgentId)

  const handleSendRequest = async (payload: RequestPayload) => {
    const userMessage = {
      type: "user" as const,
      content: payload.request,
      timestamp: new Date().toISOString(),
    }
    addMessage(displayAgentId, userMessage)

    setLoading(true)
    try {
      const response = await submitSupervisorRequest(payload)
      const agentResponse = {
        type: "agent" as const,
        content: response.response || "No response content.",
        timestamp: response.timestamp,
        metadata: response.metadata,
      }
      addMessage(displayAgentId, agentResponse)
    } catch (error) {
      const errorMessage = {
        type: "error" as const,
        content: error instanceof Error ? error.message : "An unknown error occurred.",
        timestamp: new Date().toISOString(),
      }
      addMessage(displayAgentId, errorMessage)
    } finally {
      setLoading(false)
    }
  }

  const renderChatWindow = () => (
    <ChatWindow
      agentId={displayAgentId}
      onToggleHistory={() => setShowHistory(!showHistory)}
      onToggleDetailPanel={() => setShowDetailPanel(!showDetailPanel)}
      onSendRequest={handleSendRequest}
    />
  )

  if (isMobile) {
    return (
      <div className="flex h-screen bg-background text-foreground">
        <div className="flex-1 flex flex-col">
          {renderChatWindow()}
          <Sheet open={showHistory} onOpenChange={setShowHistory}>
            <SheetContent side="right" className="w-full max-w-md p-0">
              <HistoryPanel agentId={displayAgentId} onClose={() => setShowHistory(false)} />
            </SheetContent>
          </Sheet>
          <Sheet open={showDetailPanel} onOpenChange={setShowDetailPanel}>
            <SheetContent side="left" className="w-full max-w-md p-0">
              {displayAgent && <AgentDetailPanel agent={displayAgent} onClose={() => setShowDetailPanel(false)} />}
            </SheetContent>
          </Sheet>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-background text-foreground">
      {showDetailPanel && displayAgent && (
        <div className="w-96 border-r border-border">
          <AgentDetailPanel agent={displayAgent} />
        </div>
      )}
      <div className="flex-1 flex flex-col">{renderChatWindow()}</div>
      {showHistory && (
        <div className="w-96 border-l border-border">
          <HistoryPanel agentId={displayAgentId} />
        </div>
      )}
    </div>
  )
}
