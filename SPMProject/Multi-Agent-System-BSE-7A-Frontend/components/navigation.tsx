"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Button } from "@/components/ui/button"

export function Navigation() {
  const pathname = usePathname()

  const links = [
    { href: "/dashboard", label: "Dashboard" },
    { href: "/agents", label: "Agents" },
    { href: "/admin", label: "Admin" },
  ]

  return (
    <nav className="flex gap-2">
      {links.map((link) => (
        <Link key={link.href} href={link.href}>
          <Button variant={pathname === link.href ? "default" : "ghost"} size="sm">
            {link.label}
          </Button>
        </Link>
      ))}
    </nav>
  )
}
