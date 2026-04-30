import { create } from 'zustand'

interface ToastItem {
  id: number
  message: string
  type: 's' | 'e' | 'i'
}

interface ToastStore {
  items: ToastItem[]
  add: (message: string, type?: 's' | 'e' | 'i') => void
  remove: (id: number) => void
}

let nextId = 0

export const useToast = create<ToastStore>((set) => ({
  items: [],
  add: (message, type = 'i') => {
    const id = ++nextId
    set((s) => ({ items: [...s.items, { id, message, type }] }))
    setTimeout(() => set((s) => ({ items: s.items.filter((t) => t.id !== id) })), 3500)
  },
  remove: (id) => set((s) => ({ items: s.items.filter((t) => t.id !== id) })),
}))

const typeStyles: Record<string, string> = {
  s: 'border-l-gn bg-white text-gn',
  e: 'border-l-rd bg-white text-rd',
  i: 'border-l-ac bg-white text-ac',
}

export default function ToastContainer() {
  const items = useToast((s) => s.items)

  if (!items.length) return null

  return (
    <div className="fixed top-[52px] right-3 z-[999] flex flex-col gap-1 pointer-events-none">
      {items.map((t) => (
        <div
          key={t.id}
          className={`px-3.5 py-2 rounded-lg border-l-4 text-sm pointer-events-auto shadow-elevated animate-slide-in-right
            ${typeStyles[t.type]}`}
        >
          {t.message}
        </div>
      ))}
    </div>
  )
}
