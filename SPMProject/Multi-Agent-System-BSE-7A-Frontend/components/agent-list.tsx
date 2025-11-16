"use client"
import { useAgents } from "@/context/agent-context"
import AgentCard from "./agent-card"
import { Spinner } from "@/components/ui/spinner"

interface AgentListProps {
  selectedAgentId: string | null
  onSelectAgent: (id: string) => void
}

export default function AgentList({ selectedAgentId, onSelectAgent }: AgentListProps) {
  const { agents, loading, error, agentHealth } = useAgents()

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <Spinner />
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-4">
        <div className="text-sm text-destructive">Failed to load agents: {error}</div>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-2">
      {agents.map((agent) => (
        <AgentCard
          key={agent.id}
          agent={agent}
          health={agentHealth[agent.id] || "offline"}
          isSelected={selectedAgentId === agent.id}
          onSelect={() => onSelectAgent(agent.id)}
        />
      ))}
    </div>
  )
}
