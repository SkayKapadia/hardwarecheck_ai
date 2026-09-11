"use client";

import * as React from "react"
import * as PopoverPrimitives from "@radix-ui/react-popover"
import { Info } from "lucide-react"

export function InfoPopup({ content }: { content: string }) {
  return (
    <PopoverPrimitives.Root>
      <PopoverPrimitives.Trigger asChild>
        <button 
          className="text-muted-foreground hover:text-primary transition-colors focus:outline-none inline-flex items-center ml-2"
          aria-label="More information"
        >
          <Info className="w-3.5 h-3.5" />
        </button>
      </PopoverPrimitives.Trigger>
      <PopoverPrimitives.Portal>
        <PopoverPrimitives.Content
          side="top"
          sideOffset={5}
          className="z-50 max-w-xs bg-card/95 backdrop-blur border border-primary/50 text-xs font-mono p-3 shadow-[0_0_15px_rgba(34,211,238,0.2)] text-primary leading-relaxed outline-none"
        >
          {content}
          <PopoverPrimitives.Arrow className="fill-primary/50" />
        </PopoverPrimitives.Content>
      </PopoverPrimitives.Portal>
    </PopoverPrimitives.Root>
  )
}
