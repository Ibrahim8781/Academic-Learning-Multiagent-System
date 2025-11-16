"use client"

import type React from "react"
import { renderHook, act } from "@testing-library/react"
import { SettingsProvider, useSettings } from "@/context/settings-context"

describe("SettingsContext", () => {
  it("provides default settings", () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => <SettingsProvider>{children}</SettingsProvider>

    const { result } = renderHook(() => useSettings(), { wrapper })

    expect(result.current.settings.enableLTM).toBe(false)
    expect(result.current.settings.enableSTM).toBe(false)
    expect(result.current.settings.autoRoute).toBe(false)
  })

  it("updates settings", () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => <SettingsProvider>{children}</SettingsProvider>

    const { result } = renderHook(() => useSettings(), { wrapper })

    act(() => {
      result.current.updateSettings({
        enableLTM: true,
        enableSTM: true,
        autoRoute: true,
      })
    })

    expect(result.current.settings.enableLTM).toBe(true)
    expect(result.current.settings.enableSTM).toBe(true)
    expect(result.current.settings.autoRoute).toBe(true)
  })

  it("resets settings to defaults", () => {
    const wrapper = ({ children }: { children: React.ReactNode }) => <SettingsProvider>{children}</SettingsProvider>

    const { result } = renderHook(() => useSettings(), { wrapper })

    act(() => {
      result.current.updateSettings({
        enableLTM: true,
        enableSTM: true,
        autoRoute: true,
      })
    })

    act(() => {
      result.current.resetSettings()
    })

    expect(result.current.settings.enableLTM).toBe(false)
    expect(result.current.settings.enableSTM).toBe(false)
    expect(result.current.settings.autoRoute).toBe(false)
  })
})
