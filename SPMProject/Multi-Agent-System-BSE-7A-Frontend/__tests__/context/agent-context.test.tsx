"use client"

import type React from "react"
import { renderHook, act } from "@testing-library/react"
import { AgentProvider, useAgent } from "@/context/agent-context"

describe("AgentContext", () => {
  it("provides initial agent state", () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => <AgentProvider>{children}</AgentProvider>

    const { result } = renderHook(() => useAgent(), { wrapper })

    expect(result.current.agents).toEqual([])
    expect(result.current.loading).toBe(true)
  })

  it("sets agents", () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => <AgentProvider>{children}</AgentProvider>

    const { result } = renderHook(() => useAgent(), { wrapper })

    const mockAgents = [
      { id: "1", name: "Agent 1", description: "Test agent 1" },
      { id: "2", name: "Agent 2", description: "Test agent 2" },
    ]

    act(() => {
      result.current.setAgents(mockAgents)
    })

    expect(result.current.agents).toEqual(mockAgents)
  })

  it("updates agent health status", () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => <AgentProvider>{children}</AgentProvider>

    const { result } = renderHook(() => useAgent(), { wrapper })

    const mockAgents = [{ id: "1", name: "Agent 1", description: "Test agent 1" }]

    act(() => {
      result.current.setAgents(mockAgents)
      result.current.updateAgentHealth("1", "healthy")
    })

    expect(result.current.getAgentHealth("1")).toBe("healthy")
  })
})
