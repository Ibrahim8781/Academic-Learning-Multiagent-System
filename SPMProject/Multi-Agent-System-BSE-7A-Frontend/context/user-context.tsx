"use client"

import {
  createContext,
  useContext,
  useState,
  type ReactNode,
  useEffect,
  useCallback,
} from "react"
import { type User } from "@/types"
import { login as apiLogin, getCurrentUser } from "@/lib/api-service"
import { useRouter } from "next/navigation"

interface UserContextType {
  user: User | null
  login: (credentials: Record<"email" | "password", string>) => Promise<void>
  logout: () => void
  loading: boolean
  error: string | null
}

const UserContext = createContext<UserContextType | undefined>(undefined)

export function UserProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const router = useRouter()

  const checkUser = useCallback(async () => {
    const token = localStorage.getItem("jwt")
    if (token) {
      try {
        const currentUser = await getCurrentUser()
        setUser(currentUser)
      } catch (err) {
        console.error("Failed to fetch user, token might be invalid.", err)
        localStorage.removeItem("jwt")
        setUser(null)
      }
    }
    setLoading(false)
  }, [])

  useEffect(() => {
    checkUser()
  }, [checkUser])

  const login = async (credentials: Record<"email" | "password", string>) => {
    try {
      setLoading(true)
      setError(null)
      const { token, user: loggedInUser } = await apiLogin(credentials)
      localStorage.setItem("jwt", token)
      setUser(loggedInUser)
      router.push("/dashboard")
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "An unknown error occurred"
      console.error("Login failed:", errorMsg)
      setError(errorMsg)
      throw err
    } finally {
      setLoading(false)
    }
  }

  const logout = () => {
    setUser(null)
    localStorage.removeItem("jwt")
    router.push("/login")
  }

  return (
    <UserContext.Provider value={{ user, login, logout, loading, error }}>
      {children}
    </UserContext.Provider>
  )
}

export function useUser() {
  const context = useContext(UserContext)
  if (!context) {
    throw new Error("useUser must be used within UserProvider")
  }
  return context
}
