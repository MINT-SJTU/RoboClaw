import { useEffect, useMemo } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { useChatSocket } from '@/domains/chat/store/useChatSocket'
import { useHardwareStore } from '@/domains/hardware/store/useHardwareStore'
import { useI18n } from '@/i18n'
import { StatusPill } from '@/shared/ui'
import { useAuthStore } from '@/shared/lib/authStore'

/** 手机号脱敏：138****8888 */
function maskPhone(phone: string): string {
    if (phone.length !== 11) return phone
    return `${phone.slice(0, 3)}****${phone.slice(7)}`
}

/** 用户等级徽标颜色 */
function levelColor(level: string): string {
    if (level === 'admin') return '#d97706'
    if (level === 'contributor') return '#2f6fe4'
    return '#6b7a8d'
}

export default function AppHeader() {
    const location = useLocation()
    const navigate = useNavigate()
    const { connected } = useChatSocket()
    const networkInfo = useHardwareStore((state) => state.networkInfo)
    const fetchNetworkInfo = useHardwareStore((state) => state.fetchNetworkInfo)
    const { t, locale, setLocale } = useI18n()
    const { user, isLoggedIn, isChecking, logout } = useAuthStore()

    useEffect(() => {
        fetchNetworkInfo()
    }, [fetchNetworkInfo])

    const pageTitle = useMemo(() => {
        if (location.pathname.startsWith('/control')) return t('controlCenter')
        if (location.pathname.startsWith('/datasets/explorer')) return t('datasetExplorer')
        if (location.pathname.startsWith('/datasets')) return t('datasetReader')
        if (location.pathname.startsWith('/curation/datasets')) return t('datasetReader')
        if (location.pathname.startsWith('/curation/text-alignment')) return t('textAlignment')
        if (location.pathname.startsWith('/curation/quality')) return t('qualityWorkbench')
        if (location.pathname.startsWith('/curation')) return t('pipelineNav')
        if (location.pathname.startsWith('/logs')) return t('logs')
        if (location.pathname.startsWith('/settings/hardware')) return t('settingsHardware')
        if (location.pathname.startsWith('/settings/provider')) return t('settingsProvider')
        if (location.pathname.startsWith('/settings/hub')) return t('hfConfig')
        if (location.pathname.startsWith('/settings')) return t('settings')
        return 'RoboClaw'
    }, [location.pathname, t])

    /** 用户头像首字母（昵称优先，否则取手机号前3位）*/
    const avatarInitial = user
        ? (user.nickname ? user.nickname.slice(0, 1).toUpperCase() : user.phone.slice(0, 3))
        : '?'

    function handleLogout() {
        logout()
        navigate('/login', { replace: true })
    }

    return (
        <header className="app-topbar">
            <div className="app-topbar__title">
                <div className="space-y-2">
                    <Link to="/control" className="display-title text-[1.95rem] text-tx">
                        RoboClaw
                    </Link>
                    <div className="eyebrow">{pageTitle}</div>
                </div>
            </div>

            <div className="app-topbar__actions">
                {networkInfo && (
                    <div className="rounded-full bg-white/70 px-4 py-2 text-xs font-semibold uppercase tracking-[0.14em] text-tx2">
                        {networkInfo.lan_ip}:{networkInfo.port}
                    </div>
                )}

                <StatusPill active={connected}>
                    {connected ? t('connected') : t('disconnected')}
                </StatusPill>

                {/* ── 用户区域 ── */}
                {!isChecking && (
                    isLoggedIn && user ? (
                        <>
                            <div className="header-user-badge">
                                <div
                                    className="header-user-badge__avatar"
                                    style={{ background: `linear-gradient(180deg, ${levelColor(user.level)}cc, ${levelColor(user.level)})` }}
                                >
                                    {avatarInitial}
                                </div>
                                <span className="header-user-badge__phone">{maskPhone(user.phone)}</span>
                            </div>
                            <button
                                type="button"
                                onClick={handleLogout}
                                className="header-logout-btn"
                                title={t('authLogout')}
                            >
                                {t('authLogout')}
                            </button>
                        </>
                    ) : (
                        <Link to="/login" className="header-login-btn">
                            {t('authLoginPrompt')}
                        </Link>
                    )
                )}

                <button
                    onClick={() => setLocale(locale === 'zh' ? 'en' : 'zh')}
                    className="app-topbar__locale"
                >
                    {locale === 'zh' ? 'EN' : '中文'}
                </button>
            </div>
        </header>
    )
}
