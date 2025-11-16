"use client"

import type React from "react"

import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import ChatWindow from "@/components/chat-window"
import { AgentProvider } from "@/context/agent-context"
import { HistoryProvider } from "@/context/history-context"
import { SettingsProvider } from "@/context/settings-context"
import { jest } from "@jest/globals"

// Mock the API service
jest.mock("@/lib/api-service", () => ({
  fetchAgents: jest.fn(),
  submitRequest: jest.fn(),
  checkHealth: jest.fn(),
}))

const mockAgentId = "test-agent"

const renderWithProviders = (component: React.ReactNode) => {
  return render(
    <AgentProvider>
      <HistoryProvider>
        <SettingsProvider>{component}</SettingsProvider>
      </HistoryProvider>
    </AgentProvider>,
  )
}

describe("ChatWindow", () => {
  it("renders chat window with input field", () => {
    renderWithProviders(<ChatWindow agentId={mockAgentId} onToggleHistory={() => {}} />)

    expect(screen.getByPlaceholderText(/send a message/i)).toBeInTheDocument()
  })

  it("displays messages in chat", async () => {
    renderWithProviders(<ChatWindow agentId={mockAgentId} onToggleHistory={() => {}} />)

    const input = screen.getByPlaceholderText(/send a message/i) as HTMLInputElement
    fireEvent.change(input, { target: { value: "Test message" } })
    fireEvent.click(screen.getByRole("button", { name: /send/i }))

    await waitFor(() => {
      expect(screen.getByText("Test message")).toBeInTheDocument()
    })
  })

  it("calls onToggleHistory when history button is clicked", () => {
    const onToggleHistory = jest.fn()
    renderWithProviders(<ChatWindow agentId={mockAgentId} onToggleHistory={onToggleHistory} />)

    const historyButton = screen.getByRole("button", { name: /history/i })
    fireEvent.click(historyButton)

    expect(onToggleHistory).toHaveBeenCalled()
  })
})
