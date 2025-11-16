"use client"

import { createContext, useContext, useEffect, useState, type ReactNode } from "react"
import type { Agent } from "@/types"
import { fetchAgentRegistry, checkAgentHealth } from "@/lib/api-service"
import { useUser } from "./user-context"

interface AgentContextType {
  agents: Agent[]
  loading: boolean
  error: string | null
  agentHealth: Record<string, "healthy" | "degraded" | "offline">
  refreshHealth: (agentId?: string) => Promise<void>
}

const AgentContext = createContext<AgentContextType | undefined>(undefined)

export function AgentProvider({ children }: { children: ReactNode }) {
  const [agents, setAgents] = useState<Agent[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [agentHealth, setAgentHealth] = useState<Record<string, "healthy" | "degraded" | "offline">>({})
  const { user, loading: userLoading } = useUser()

  // Load agents on mount, but only when the user is authenticated
  useEffect(() => {
    const loadAgents = async () => {
      if (!user) {
        setLoading(false)
        return // Don't fetch if there's no user
      }
      try {
        setLoading(true)
        const data = await fetchAgentRegistry()
        setAgents(data)
        setError(null)

        // Initial health check
        const healthMap: Record<string, "healthy" | "degraded" | "offline"> = {}
        for (const agent of data) {
          healthMap[agent.id] = await checkAgentHealth(agent)
        }
        setAgentHealth(healthMap)
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : "Failed to load agents"
        console.error("[AgentContext] Error loading agents:", errorMsg)
        setError(errorMsg)
        setAgents([])
      } finally {
        setLoading(false)
      }
    }

    if (!userLoading) {
      loadAgents()
    }
  }, [user, userLoading])

  // Poll health status every 30 seconds
  useEffect(() => {
    if (!user || agents.length === 0) return // Don't poll if no user or no agents

    const pollHealth = async () => {
      const healthMap: Record<string, "healthy" | "degraded" | "offline"> = {}
      for (const agent of agents) {
        healthMap[agent.id] = await checkAgentHealth(agent)
      }
      setAgentHealth(healthMap)
    }

    const interval = setInterval(pollHealth, 30000)

    return () => clearInterval(interval)
  }, [agents, user])

  const refreshHealth = async (agentId?: string) => {
    if (!user) return // Can't refresh health without a user

    if (agentId) {
      const agentToRefresh = agents.find((a: Agent) => a.id === agentId)
      if (agentToRefresh) {
        const status = await checkAgentHealth(agentToRefresh)
        setAgentHealth((prev: Record<string, "healthy" | "degraded" | "offline">) => ({ ...prev, [agentId]: status }))
      }
    } else {
      const healthMap: Record<string, "healthy" | "degraded" | "offline"> = {}
      for (const agent of agents) {
        healthMap[agent.id] = await checkAgentHealth(agent)
      }
      setAgentHealth(healthMap)
    }
  }

  return (
    <AgentContext.Provider value={{ agents, loading, error, agentHealth, refreshHealth }}>
      {children}
    </AgentContext.Provider>
  )
}

export function useAgents() {
  const context = useContext(AgentContext)
  if (!context) {
    throw new Error("useAgents must be used within AgentProvider")
  }
  return context
}

export const useAgent = useAgents
