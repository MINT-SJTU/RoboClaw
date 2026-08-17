/**
 * 认证状态管理（Zustand store）
 *
 * 设计原则：
 * - 登录状态完全基于云端 ECS 后端，必须联网才能维持
 * - isChecking 期间应用正常渲染，不阻塞本地功能
 * - 网络不可达时 cloudAvailable=false，本地功能不受影响
 */

import { create } from 'zustand'
import { evoApi, type UserInfo } from '@/shared/api/evoClient'

const ACCESS_KEY = 'evo_access_token'
const REFRESH_KEY = 'evo_refresh_token'

interface AuthState {
    user: UserInfo | null
    isLoggedIn: boolean
    /** 应用启动时正在异步验证 token，期间为 true */
    isChecking: boolean
    /** 云端后端是否可达（网络层面）*/
    cloudAvailable: boolean

    /** 应用启动时调用一次，异步验证本地 token */
    initialize: () => Promise<void>
    /** 登录成功后，将 token 写入 localStorage */
    setTokens: (access: string, refresh: string) => void
    /** 登录成功后，设置用户信息 */
    setUser: (user: UserInfo) => void
    /** 退出登录 */
    logout: () => void
}

export const useAuthStore = create<AuthState>((set) => ({
    user: null,
    isLoggedIn: false,
    isChecking: true,
    cloudAvailable: false,

    initialize: async () => {
        const accessToken = localStorage.getItem(ACCESS_KEY)

        if (!accessToken) {
            set({ isChecking: false })
            return
        }

        try {
            const user = await evoApi.getMe()
            set({ user, isLoggedIn: true, isChecking: false, cloudAvailable: true })
        } catch (err: unknown) {
            // access_token 失效，尝试 refresh
            const refreshToken = localStorage.getItem(REFRESH_KEY)
            if (refreshToken) {
                try {
                    const tokens = await evoApi.refresh(refreshToken)
                    localStorage.setItem(ACCESS_KEY, tokens.access_token)
                    localStorage.setItem(REFRESH_KEY, tokens.refresh_token)
                    const user = await evoApi.getMe()
                    set({ user, isLoggedIn: true, isChecking: false, cloudAvailable: true })
                    return
                } catch (_) {
                    localStorage.removeItem(ACCESS_KEY)
                    localStorage.removeItem(REFRESH_KEY)
                }
            }

            // TypeError 通常是网络不通（fetch failed），其他是 401/403 等
            const isNetworkError = err instanceof TypeError
            set({ isChecking: false, cloudAvailable: !isNetworkError })
        }
    },

    setTokens: (access, refresh) => {
        localStorage.setItem(ACCESS_KEY, access)
        localStorage.setItem(REFRESH_KEY, refresh)
    },

    setUser: (user) => {
        set({ user, isLoggedIn: true, cloudAvailable: true })
    },

    logout: () => {
        localStorage.removeItem(ACCESS_KEY)
        localStorage.removeItem(REFRESH_KEY)
        set({ user: null, isLoggedIn: false })
    },
}))
