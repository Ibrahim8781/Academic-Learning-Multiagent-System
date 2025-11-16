"use client"

import { useParams } from "next/navigation"
import Link from "next/link"
import Header from "@/components/header"
import { useAgents } from "@/context/agent-context"
import { useHistory } from "@/context/history-context"
import AgentDetailPanel from "@/components/agent-detail-panel"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { LoadingSpinner } from "@/components/loading-spinner"

export default function AgentDetailPage() {
  const params = useParams()
  const agentId = params.id as string
  const { agents, loading } = useAgents()
  const { getHistory } = useHistory()

  const agent = agents.find((a) => a.id === agentId)
  const history = getHistory(agentId)

  if (loading) {
    return (
      <div className="flex h-screen bg-background text-foreground">
        <div className="flex-1 flex flex-col">
          <Header />
          <div className="flex-1 flex items-center justify-center">
            <LoadingSpinner />
          </div>
        </div>
      </div>
    )
  }

  if (!agent) {
    return (
      <div className="flex h-screen bg-background text-foreground">
        <div className="flex-1 flex flex-col">
          <Header />
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <h2 className="text-2xl font-bold mb-2">Agent Not Found</h2>
              <Link href="/agents">
                <Button>Back to Directory</Button>
              </Link>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-background text-foreground">
      <div className="flex-1 flex flex-col">
        <Header />
        <div className="flex-1 overflow-auto p-6">
          <div className="max-w-6xl mx-auto space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold mb-2">{agent.name}</h1>
                <p className="text-muted-foreground">{agent.description}</p>
              </div>
              <Link href="/agents">
                <Button variant="outline">Back to Directory</Button>
              </Link>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Main Content */}
              <div className="lg:col-span-2 space-y-6">
                {/* Capabilities */}
                <Card className="p-6">
                  <h2 className="text-xl font-semibold mb-4">Capabilities</h2>
                  <div className="flex flex-wrap gap-2">
                    {agent.capabilities?.map((cap) => (
                      <span key={cap} className="px-3 py-1 rounded-full bg-primary/10 text-primary text-sm font-medium">
                        {cap}
                      </span>
                    ))}
                  </div>
                </Card>

                {/* Recent Interactions */}
                <Card className="p-6">
                  <h2 className="text-xl font-semibold mb-4">Last 10 Interactions</h2>
                  {history.length === 0 ? (
                    <p className="text-muted-foreground">No interactions yet</p>
                  ) : (
                    <div className="space-y-3">
                      {history.slice(-10).map((msg, idx) => (
                        <div key={idx} className="p-3 rounded-lg border border-border">
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <p className="text-sm font-medium capitalize">{msg.type}</p>
                              <p className="text-sm text-muted-foreground line-clamp-2">{msg.content}</p>
                            </div>
                            <span className="text-xs text-muted-foreground ml-2">
                              {new Date(msg.timestamp).toLocaleString()}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </Card>
              </div>

              {/* Sidebar */}
              <div>
                <AgentDetailPanel agent={agent} onClose={() => {}} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
