"use client"

import { render, screen, fireEvent } from "@testing-library/react"
import RequestComposer from "@/components/request-composer"
import { jest } from "@jest/globals"

describe("RequestComposer", () => {
  it("renders input field and send button", () => {
    const onSubmit = jest.fn()
    render(<RequestComposer onSubmit={onSubmit} isLoading={false} />)

    expect(screen.getByPlaceholderText(/send a message/i)).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /send/i })).toBeInTheDocument()
  })

  it("calls onSubmit with message content", () => {
    const onSubmit = jest.fn()
    render(<RequestComposer onSubmit={onSubmit} isLoading={false} />)

    const input = screen.getByPlaceholderText(/send a message/i) as HTMLInputElement
    fireEvent.change(input, { target: { value: "Test message" } })
    fireEvent.click(screen.getByRole("button", { name: /send/i }))

    expect(onSubmit).toHaveBeenCalledWith("Test message")
    expect(input.value).toBe("")
  })

  it("disables send button when loading", () => {
    const onSubmit = jest.fn()
    render(<RequestComposer onSubmit={onSubmit} isLoading={true} />)

    const button = screen.getByRole("button", { name: /send/i })
    expect(button).toBeDisabled()
  })

  it("prevents empty message submission", () => {
    const onSubmit = jest.fn()
    render(<RequestComposer onSubmit={onSubmit} isLoading={false} />)

    fireEvent.click(screen.getByRole("button", { name: /send/i }))
    expect(onSubmit).not.toHaveBeenCalled()
  })
})
