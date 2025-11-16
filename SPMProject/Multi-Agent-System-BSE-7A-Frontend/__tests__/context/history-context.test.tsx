import type React from "react"
import { renderHook, act } from "@testing-library/react"
import { HistoryProvider, useHistory } from "@/context/history-context"

describe("HistoryContext", () => {
  it("adds and retrieves messages", () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => <HistoryProvider>{children}</HistoryProvider>

    const { result } = renderHook(() => useHistory(), { wrapper })

    act(() => {
      result.current.addMessage("agent-1", {
        type: "user",
        content: "Hello",
        timestamp: new Date().toISOString(),
      })
    })

    const history = result.current.getHistory("agent-1")
    expect(history).toHaveLength(1)
    expect(history[0].content).toBe("Hello")
  })

  it("clears history for an agent", () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => <HistoryProvider>{children}</HistoryProvider>

    const { result } = renderHook(() => useHistory(), { wrapper })

    act(() => {
      result.current.addMessage("agent-1", {
        type: "user",
        content: "Hello",
        timestamp: new Date().toISOString(),
      })
    })

    act(() => {
      result.current.clearHistory("agent-1")
    })

    const history = result.current.getHistory("agent-1")
    expect(history).toHaveLength(0)
  })
})
