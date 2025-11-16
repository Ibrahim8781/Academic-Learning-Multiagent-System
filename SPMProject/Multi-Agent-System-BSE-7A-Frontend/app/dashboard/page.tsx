"use client"

import { useState } from "react"
import Link from "next/link"
import Header from "@/components/header"
import { useAgents } from "@/context/agent-context"
import { useHistory } from "@/context/history-context"
import RequestComposer from "@/components/request-composer"
import HealthStatus from "@/components/health-status"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"

export default function DashboardPage() {
  const { agents, loading, agentHealth } = useAgents()
  const { getHistory } = useHistory()
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null)

  // Get recent history across all agents
  const recentHistory = agents
    .flatMap((agent) => {
      const history = getHistory(agent.id)
      return history.slice(-3).map((msg) => ({
        ...msg,
        agentId: agent.id,
        agentName: agent.name,
      }))
    })
    .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
    .slice(0, 5)

  // Get active agents (healthy status)
  const activeAgents = agents.slice(0, 3)

  return (
    <div className="flex h-screen bg-background text-foreground">
      <div className="flex-1 flex flex-col">
        <Header />
        <div className="flex-1 overflow-auto p-6">
          <div className="max-w-6xl mx-auto space-y-6">
            {/* Header */}
            <div>
              <h1 className="text-3xl font-bold mb-2">Dashboard</h1>
              <p className="text-muted-foreground">Quick access to agents and recent activity</p>
            </div>

            {/* Quick Composer */}
            <Card className="p-6">
              <h2 className="text-xl font-semibold mb-4">Quick Request</h2>
              <RequestComposer agentId={selectedAgentId || agents[0]?.id} onAgentChange={setSelectedAgentId} />
            </Card>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Active Agents */}
              <div className="lg:col-span-2">
                <Card className="p-6">
                  <h2 className="text-xl font-semibold mb-4">Active Agents</h2>
                  {loading ? (
                    <p className="text-muted-foreground">Loading agents...</p>
                  ) : (
                    <div className="space-y-3">
                      {activeAgents.map((agent) => (
                        <Link
                          key={agent.id}
                          href={`/conversation/${agent.id}`}
                          className="block p-3 rounded-lg border border-border hover:bg-muted transition-colors"
                        >
                          <div className="flex items-center justify-between">
                            <div>
                              <h3 className="font-medium">{agent.name}</h3>
                              <p className="text-sm text-muted-foreground">{agent.description}</p>
                            </div>
                            <HealthStatus status={agentHealth[agent.id] || "healthy"} />
                          </div>
                        </Link>
                      ))}
                    </div>
                  )}
                  <Link href="/agents">
                    <Button variant="outline" className="w-full mt-4 bg-transparent">
                      View All Agents
                    </Button>
                  </Link>
                </Card>
              </div>

              {/* Aggregated Health */}
              <div>
                <Card className="p-6">
                  <h2 className="text-xl font-semibold mb-4">System Health</h2>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Healthy</span>
                      <span className="font-semibold text-green-600">
                        {Object.values(agentHealth).filter((s) => s === "healthy").length}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Degraded</span>
                      <span className="font-semibold text-yellow-600">
                        {Object.values(agentHealth).filter((s) => s === "degraded").length}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Offline</span>
                      <span className="font-semibold text-red-600">
                        {Object.values(agentHealth).filter((s) => s === "offline").length}
                      </span>
                    </div>
                  </div>
                </Card>
              </div>
            </div>

            {/* Recent History */}
            {recentHistory.length > 0 && (
              <Card className="p-6">
                <h2 className="text-xl font-semibold mb-4">Recent Activity</h2>
                <div className="space-y-3">
                  {recentHistory.map((item, idx) => (
                    <div key={idx} className="p-3 rounded-lg border border-border">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <p className="text-sm font-medium">{item.agentName}</p>
                          <p className="text-sm text-muted-foreground line-clamp-2">{item.content}</p>
                        </div>
                        <span className="text-xs text-muted-foreground ml-2">
                          {new Date(item.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
