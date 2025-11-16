"use client"

import type React from "react"
import { AgentProvider } from "@/context/agent-context"
import { HistoryProvider } from "@/context/history-context"
import { SettingsProvider } from "@/context/settings-context"
import { UserProvider } from "@/context/user-context"

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <UserProvider>
      <AgentProvider>
        <HistoryProvider>
          <SettingsProvider>{children}</SettingsProvider>
        </HistoryProvider>
      </AgentProvider>
    </UserProvider>
  )
}
