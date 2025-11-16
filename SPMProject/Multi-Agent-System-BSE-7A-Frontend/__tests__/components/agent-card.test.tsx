"use client"

import { render, screen } from "@testing-library/react"
import AgentCard from "@/components/agent-card"
import { jest } from "@jest/globals"

describe("AgentCard", () => {
  const mockAgent = {
    id: "test-agent",
    name: "Test Agent",
    description: "A test agent",
  }

  it("renders agent information", () => {
    render(<AgentCard agent={mockAgent} isSelected={false} onSelect={() => {}} />)

    expect(screen.getByText("Test Agent")).toBeInTheDocument()
    expect(screen.getByText("A test agent")).toBeInTheDocument()
  })

  it("applies selected styles when selected", () => {
    const { container } = render(<AgentCard agent={mockAgent} isSelected={true} onSelect={() => {}} />)

    const button = container.querySelector("button")
    expect(button).toHaveClass("bg-primary")
  })

  it("calls onSelect when clicked", () => {
    const onSelect = jest.fn()
    render(<AgentCard agent={mockAgent} isSelected={false} onSelect={onSelect} />)

    screen.getByRole("button").click()
    expect(onSelect).toHaveBeenCalled()
  })
})
