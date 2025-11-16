"use client"

import { createContext, useContext, useState, type ReactNode, useEffect } from "react"
import { useUser } from "./user-context"

interface Settings {
  enableLTM: boolean
  enableSTM: boolean
  autoRoute: boolean
}

interface SettingsContextType {
  settings: Settings
  updateSettings: (settings: Partial<Settings>) => void
}

const SettingsContext = createContext<SettingsContextType | undefined>(undefined)

const DEFAULT_SETTINGS: Settings = {
  enableLTM: true,
  enableSTM: true,
  autoRoute: false,
}

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS)
  const { user } = useUser()

  const getSettingsKey = () => (user ? `agent-settings-${user.id}` : null)

  useEffect(() => {
    const settingsKey = getSettingsKey()
    if (!settingsKey) {
      setSettings(DEFAULT_SETTINGS)
      return
    }
    const stored = localStorage.getItem(settingsKey)
    if (stored) {
      try {
        setSettings(JSON.parse(stored))
      } catch {
        console.error("Failed to load settings from localStorage")
      }
    } else {
      setSettings(DEFAULT_SETTINGS)
    }
  }, [user])

  const updateSettings = (newSettings: Partial<Settings>) => {
    const updated = { ...settings, ...newSettings }
    setSettings(updated)
    const settingsKey = getSettingsKey()
    if (settingsKey) {
      localStorage.setItem(settingsKey, JSON.stringify(updated))
    }
  }

  return <SettingsContext.Provider value={{ settings, updateSettings }}>{children}</SettingsContext.Provider>
}

export function useSettings() {
  const context = useContext(SettingsContext)
  if (!context) {
    throw new Error("useSettings must be used within SettingsProvider")
  }
  return context
}
