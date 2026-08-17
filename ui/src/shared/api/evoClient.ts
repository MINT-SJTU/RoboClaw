/**
 * EvoMind 云端后端 API 客户端
 *
 * 所有用户认证均通过此文件访问阿里云 ECS 上的 evo-data 后端。
 * 本地 RoboClaw 后端（8766）的请求不经过此文件。
 *
 * 配置：在 ui/.env 或 ui/.env.local 中设置
 *   VITE_EVO_API_URL=https://api.evomind-tech.com
 */

const EVO_API = (import.meta.env.VITE_EVO_API_URL as string | undefined) ?? 'https://api.evomind-tech.com'

const ACCESS_KEY = 'evo_access_token'

// ─── Types ────────────────────────────────────────────────────────────────────

export interface CaptchaResponse {
    captcha_id: string
    image_base64: string
}

export interface TokenResponse {
    access_token: string
    refresh_token: string
    token_type: string
}

export interface UserInfo {
    id: string
    phone: string
    nickname: string | null
    level: 'normal' | 'contributor' | 'admin'
    rank: number
    has_password: boolean
    created_at: string
}

// ─── Core request ─────────────────────────────────────────────────────────────

async function evoRequest<T>(
    path: string,
    options: RequestInit = {},
    withAuth = false,
): Promise<T> {
    const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        ...(options.headers as Record<string, string>),
    }

    if (withAuth) {
        const token = localStorage.getItem(ACCESS_KEY)
        if (token) headers['Authorization'] = `Bearer ${token}`
    }

    const res = await fetch(`${EVO_API}${path}`, { ...options, headers })

    if (!res.ok) {
        let detail = `HTTP ${res.status}`
        try {
            const body = await res.json()
            detail = body.detail || detail
        } catch (_) { /* ignore */ }
        throw new Error(detail)
    }

    if (res.status === 204) return undefined as T
    return res.json() as Promise<T>
}

// ─── Auth API ─────────────────────────────────────────────────────────────────

export const evoApi = {
    /** 获取图形验证码 */
    getCaptcha: (): Promise<CaptchaResponse> =>
        evoRequest('/auth/captcha'),

    /** 发送短信验证码（登录场景）*/
    sendSms: (phone: string, captchaId: string, captchaText: string): Promise<void> =>
        evoRequest('/auth/send_sms', {
            method: 'POST',
            body: JSON.stringify({ phone, captcha_id: captchaId, captcha_text: captchaText, scene: 'login' }),
        }),

    /** 发送短信验证码（指定场景：login / change_phone / reset_password）*/
    sendSmsWithScene: (
        phone: string,
        captchaId: string,
        captchaText: string,
        scene: 'login' | 'change_phone' | 'reset_password',
    ): Promise<void> =>
        evoRequest('/auth/send_sms', {
            method: 'POST',
            body: JSON.stringify({ phone, captcha_id: captchaId, captcha_text: captchaText, scene }),
        }),

    /** 更换绑定手机号（需要 access_token + 新号码 + 短信验证码）*/
    changePhone: (newPhone: string, smsCode: string): Promise<void> =>
        evoRequest('/auth/change-phone', {
            method: 'POST',
            body: JSON.stringify({ new_phone: newPhone, sms_code: smsCode }),
        }, true),

    /** 重置/设置登录密码（手机号 + 短信验证码 + 新密码）*/
    resetPassword: (phone: string, smsCode: string, newPassword: string): Promise<void> =>
        evoRequest('/auth/reset-password', {
            method: 'POST',
            body: JSON.stringify({ phone, sms_code: smsCode, new_password: newPassword }),
        }),

    /** 手机号 + 短信验证码登录（无账号则自动注册）*/
    login: (phone: string, smsCode: string): Promise<TokenResponse> =>
        evoRequest('/auth/login', {
            method: 'POST',
            body: JSON.stringify({ phone, sms_code: smsCode }),
        }),

    /** 手机号 + 密码 + 图形验证码登录 */
    loginWithPassword: (
        phone: string,
        password: string,
        captchaId: string,
        captchaText: string,
    ): Promise<TokenResponse> =>
        evoRequest('/auth/login/password', {
            method: 'POST',
            body: JSON.stringify({ phone, password, captcha_id: captchaId, captcha_text: captchaText }),
        }),

    /** 用 refresh_token 换新 token 对 */
    refresh: (refreshToken: string): Promise<TokenResponse> =>
        evoRequest('/auth/refresh', {
            method: 'POST',
            body: JSON.stringify({ refresh_token: refreshToken }),
        }),

    /** 获取当前登录用户信息（需要 access_token）*/
    getMe: (): Promise<UserInfo> =>
        evoRequest('/auth/me', {}, true),

    /** 修改昵称 */
    updateNickname: (nickname: string): Promise<UserInfo> =>
        evoRequest('/auth/me/nickname', {
            method: 'PATCH',
            body: JSON.stringify({ nickname }),
        }, true),
}
