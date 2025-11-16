"use client"

import { useUser } from "@/context/user-context"
import { usePathname, useRouter } from "next/navigation"
import { useEffect, type ReactNode } from "react"
import { LoadingSpinner } from "@/components/loading-spinner"

export default function AuthGuard({ children }: { children: ReactNode }) {
  const { user, loading } = useUser()
  const router = useRouter()
  const pathname = usePathname()

  useEffect(() => {
    if (!loading && !user && pathname !== "/login") {
      router.push("/login")
    }
  }, [user, loading, router, pathname])

  if (loading || (!user && pathname !== "/login")) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <LoadingSpinner />
      </div>
    )
  }

  return <>{children}</>
}
