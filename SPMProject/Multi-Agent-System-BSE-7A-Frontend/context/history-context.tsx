"use client"

import { createContext, useContext, useState, type ReactNode, useEffect } from "react"
import { useUser } from "./user-context"

import { type Message } from "@/types"

interface HistoryContextType {
  getHistory: (agentId: string) => Message[]
  addMessage: (agentId: string, message: Message) => void
  clearHistory: (agentId: string) => void
}

const HistoryContext = createContext<HistoryContextType | undefined>(undefined)

const EMPTY_HISTORY: Message[] = []

export function HistoryProvider({ children }: { children: ReactNode }) {
  const [history, setHistory] = useState<Record<string, Message[]>>({})
  const { user } = useUser()

  const getHistoryKey = () => (user ? `agent-history-${user.id}` : null)

  // Load from localStorage on mount & user change
  useEffect(() => {
    const historyKey = getHistoryKey()
    if (!historyKey) {
      setHistory({}) // Clear history if no user
      return
    }
    const stored = localStorage.getItem(historyKey)
    if (stored) {
      try {
        setHistory(JSON.parse(stored))
      } catch {
        console.error("Failed to load history from localStorage")
      }
    } else {
      setHistory({}) // No history for this user yet
    }
  }, [user])

  // Save to localStorage whenever history changes
  useEffect(() => {
    const historyKey = getHistoryKey()
    if (historyKey) {
      localStorage.setItem(historyKey, JSON.stringify(history))
    }
  }, [history, user])

  const getHistory = (agentId: string) => history[agentId] || EMPTY_HISTORY

  const addMessage = (agentId: string, message: Message) => {
    setHistory((prev) => ({
      ...prev,
      [agentId]: [...(prev[agentId] || []), message],
    }))
  }

  const clearHistory = (agentId: string) => {
    setHistory((prev) => ({
      ...prev,
      [agentId]: [],
    }))
  }

  return <HistoryContext.Provider value={{ getHistory, addMessage, clearHistory }}>{children}</HistoryContext.Provider>
}

export function useHistory() {
  const context = useContext(HistoryContext)
  if (!context) {
    throw new Error("useHistory must be used within HistoryProvider")
  }
  return context
}
