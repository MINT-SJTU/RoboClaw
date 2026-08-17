import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { evoApi } from '@/shared/api/evoClient'
import { useAuthStore } from '@/shared/lib/authStore'
import { useI18n } from '@/i18n'

type LoginMode = 'sms' | 'password'
type Step = 'phone' | 'sms'

// ─── Sub-components ───────────────────────────────────────────────────────────

function PhoneIcon() {
    return (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <rect x="5" y="2" width="14" height="20" rx="2" />
            <line x1="12" y1="18" x2="12.01" y2="18" />
        </svg>
    )
}

function LockIcon() {
    return (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="11" width="18" height="11" rx="2" />
            <path d="M7 11V7a5 5 0 0 1 10 0v4" />
        </svg>
    )
}

function EyeIcon({ open }: { open: boolean }) {
    return open ? (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
            <circle cx="12" cy="12" r="3" />
        </svg>
    ) : (
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
            <line x1="1" y1="1" x2="23" y2="23" />
        </svg>
    )
}

function RefreshIcon() {
    return (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="23 4 23 10 17 10" />
            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
        </svg>
    )
}

function Spinner() {
    return (
        <svg className="animate-spin" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" strokeOpacity="0.25" />
            <path d="M12 2a10 10 0 0 1 10 10" />
        </svg>
    )
}

// ─── Input Field ──────────────────────────────────────────────────────────────

function InputField({
    icon,
    type = 'text',
    placeholder,
    value,
    onChange,
    disabled,
    maxLength,
    autoFocus,
    rightSlot,
}: {
    icon?: React.ReactNode
    type?: string
    placeholder: string
    value: string
    onChange: (v: string) => void
    disabled?: boolean
    maxLength?: number
    autoFocus?: boolean
    rightSlot?: React.ReactNode
}) {
    return (
        <div className="relative flex items-center">
            {icon && (
                <span className="pointer-events-none absolute left-3.5 text-[color:var(--tx2)]">
                    {icon}
                </span>
            )}
            <input
                type={type}
                placeholder={placeholder}
                value={value}
                onChange={(e) => onChange(e.target.value)}
                disabled={disabled}
                maxLength={maxLength}
                autoFocus={autoFocus}
                className="w-full rounded-xl border border-[color:var(--bd)] bg-white/80 px-4 py-3 text-sm text-[color:var(--tx)] placeholder-[color:var(--tx2)] outline-none transition focus:border-[color:var(--ac)] focus:ring-2 focus:ring-[rgba(47,111,228,0.15)] disabled:opacity-50"
                style={{ paddingLeft: icon ? '2.5rem' : undefined, paddingRight: rightSlot ? '7rem' : undefined }}
            />
            {rightSlot && (
                <div className="absolute right-1.5">
                    {rightSlot}
                </div>
            )}
        </div>
    )
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function LoginPage() {
    const navigate = useNavigate()
    const { setTokens, setUser, isLoggedIn } = useAuthStore()
    const { t, locale, setLocale } = useI18n()

    const [mode, setMode] = useState<LoginMode>('sms')
    const [step, setStep] = useState<Step>('phone')

    const [phone, setPhone] = useState('')
    const [captchaId, setCaptchaId] = useState('')
    const [captchaImg, setCaptchaImg] = useState('')
    const [captchaText, setCaptchaText] = useState('')
    const [smsCode, setSmsCode] = useState('')
    const [password, setPassword] = useState('')
    const [showPassword, setShowPassword] = useState(false)

    const [countdown, setCountdown] = useState(0)
    const [loading, setLoading] = useState(false)
    const [captchaLoading, setCaptchaLoading] = useState(false)
    const [error, setError] = useState('')
    const [agreed, setAgreed] = useState(false)
    const [showTerms, setShowTerms] = useState(false)
    const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

    // 已登录跳转
    useEffect(() => {
        if (isLoggedIn) navigate('/', { replace: true })
    }, [isLoggedIn, navigate])

    // 初始获取图形验证码
    useEffect(() => {
        void fetchCaptcha()
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    useEffect(() => {
        return () => { if (timerRef.current) clearInterval(timerRef.current) }
    }, [])

    async function fetchCaptcha() {
        setCaptchaLoading(true)
        setCaptchaText('')
        try {
            const res = await evoApi.getCaptcha()
            setCaptchaId(res.captcha_id)
            setCaptchaImg(res.image_base64)
        } catch {
            setError(t('authCaptchaFetchError'))
        } finally {
            setCaptchaLoading(false)
        }
    }

    function switchMode(m: LoginMode) {
        setMode(m)
        setStep('phone')
        setError('')
        setSmsCode('')
        setPassword('')
        void fetchCaptcha()
    }

    function startCountdown() {
        setCountdown(60)
        if (timerRef.current) clearInterval(timerRef.current)
        timerRef.current = setInterval(() => {
            setCountdown((c) => {
                if (c <= 1) { clearInterval(timerRef.current!); return 0 }
                return c - 1
            })
        }, 1000)
    }

    // ── 发送短信验证码 ──────────────────────────────────────────────────────────

    async function handleSendSms() {
        setError('')
        if (!agreed) { setError(t('authAgreedRequired')); return }
        if (!phone || phone.length !== 11) { setError(t('authPhoneError')); return }
        if (!captchaText) { setError(t('authCaptchaRequired')); return }

        setLoading(true)
        try {
            await evoApi.sendSms(phone, captchaId, captchaText)
            setStep('sms')
            startCountdown()
        } catch (e: unknown) {
            setError((e as Error).message || t('authSendFailed'))
            void fetchCaptcha()
        } finally {
            setLoading(false)
        }
    }

    // ── 短信验证码登录 ──────────────────────────────────────────────────────────

    async function handleSmsLogin() {
        setError('')
        if (!smsCode || smsCode.length !== 6) { setError(t('authSmsCodeError')); return }

        setLoading(true)
        try {
            const tokens = await evoApi.login(phone, smsCode)
            setTokens(tokens.access_token, tokens.refresh_token)
            const user = await evoApi.getMe()
            setUser(user)
            navigate('/', { replace: true })
        } catch (e: unknown) {
            setError((e as Error).message || t('authLoginFailed'))
        } finally {
            setLoading(false)
        }
    }

    // ── 密码登录 ────────────────────────────────────────────────────────────────

    async function handlePasswordLogin() {
        setError('')
        if (!agreed) { setError(t('authAgreedRequired')); return }
        if (!phone || phone.length !== 11) { setError(t('authPhoneError')); return }
        if (!password) { setError(t('authPasswordRequired')); return }
        if (!captchaText) { setError(t('authCaptchaRequired')); return }

        setLoading(true)
        try {
            const tokens = await evoApi.loginWithPassword(phone, password, captchaId, captchaText)
            setTokens(tokens.access_token, tokens.refresh_token)
            const user = await evoApi.getMe()
            setUser(user)
            navigate('/', { replace: true })
        } catch (e: unknown) {
            setError((e as Error).message || t('authLoginFailed'))
            void fetchCaptcha()
        } finally {
            setLoading(false)
        }
    }

    function handleSubmit(e: React.FormEvent) {
        e.preventDefault()
        if (mode === 'password') {
            void handlePasswordLogin()
        } else if (step === 'phone') {
            void handleSendSms()
        } else {
            void handleSmsLogin()
        }
    }

    // ─── Render ──────────────────────────────────────────────────────────────────

    return (
        <div className="login-page">
            {/* 语言切换 */}
            <button
                type="button"
                onClick={() => setLocale(locale === 'zh' ? 'en' : 'zh')}
                className="login-page__locale"
            >
                {locale === 'zh' ? 'EN' : '中文'}
            </button>

            <div className="login-page__card">
                {/* Logo & Brand */}
                <div className="login-page__brand">
                    <div className="login-page__logo">
                        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M12 2L2 7l10 5 10-5-10-5z" />
                            <path d="M2 17l10 5 10-5" />
                            <path d="M2 12l10 5 10-5" />
                        </svg>
                    </div>
                    <div>
                        <div className="login-page__brand-name">RoboClaw</div>
                        <div className="login-page__brand-sub">{t('authLoginTitle')}</div>
                    </div>
                </div>

                {/* Mode Tabs */}
                <div className="login-page__tabs">
                    <button
                        type="button"
                        className={`login-page__tab${mode === 'sms' ? ' login-page__tab--active' : ''}`}
                        onClick={() => switchMode('sms')}
                    >
                        {t('authModeSms')}
                    </button>
                    <button
                        type="button"
                        className={`login-page__tab${mode === 'password' ? ' login-page__tab--active' : ''}`}
                        onClick={() => switchMode('password')}
                    >
                        {t('authModePassword')}
                    </button>
                </div>

                {/* Form */}
                <form onSubmit={handleSubmit} className="login-page__form" noValidate>

                    {/* Step 1: phone + captcha (SMS mode) / phone + captcha + password (password mode) */}
                    {(mode === 'password' || step === 'phone') && (
                        <>
                            <InputField
                                icon={<PhoneIcon />}
                                type="tel"
                                placeholder={t('authPhonePlaceholder')}
                                value={phone}
                                onChange={setPhone}
                                disabled={loading || (mode === 'sms' && step === 'sms')}
                                maxLength={11}
                                autoFocus
                            />

                            {/* Captcha row */}
                            <div className="login-page__captcha-row">
                                <InputField
                                    placeholder={t('authCaptchaPlaceholder')}
                                    value={captchaText}
                                    onChange={setCaptchaText}
                                    disabled={loading}
                                    maxLength={6}
                                />
                                <button
                                    type="button"
                                    onClick={fetchCaptcha}
                                    disabled={captchaLoading}
                                    className="login-page__captcha-img"
                                    title={t('authRefreshCaptcha')}
                                >
                                    {captchaImg
                                        ? <img
                                            src={captchaImg.startsWith('data:') ? captchaImg : `data:image/png;base64,${captchaImg}`}
                                            alt="captcha"
                                            className="h-full w-auto block"
                                        />
                                        : <div className="flex items-center justify-center text-[color:var(--tx2)]"><RefreshIcon /></div>
                                    }
                                </button>
                            </div>

                            {/* Password input (password mode only) */}
                            {mode === 'password' && (
                                <div className="relative flex items-center">
                                    <span className="pointer-events-none absolute left-3.5 text-[color:var(--tx2)]">
                                        <LockIcon />
                                    </span>
                                    <input
                                        type={showPassword ? 'text' : 'password'}
                                        placeholder={t('authPasswordPlaceholder')}
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        disabled={loading}
                                        className="w-full rounded-xl border border-[color:var(--bd)] bg-white/80 py-3 pr-11 text-sm text-[color:var(--tx)] placeholder-[color:var(--tx2)] outline-none transition focus:border-[color:var(--ac)] focus:ring-2 focus:ring-[rgba(47,111,228,0.15)] disabled:opacity-50"
                                        style={{ paddingLeft: '2.5rem' }}
                                    />
                                    <button
                                        type="button"
                                        onClick={() => setShowPassword((v) => !v)}
                                        className="absolute right-3.5 text-[color:var(--tx2)] hover:text-[color:var(--tx)]"
                                    >
                                        <EyeIcon open={showPassword} />
                                    </button>
                                </div>
                            )}
                        </>
                    )}

                    {/* Step 2: SMS code input (SMS mode only) */}
                    {mode === 'sms' && step === 'sms' && (
                        <>
                            <div className="login-page__phone-hint">
                                <span className="text-sm text-[color:var(--tx2)]">{t('authSmsCodeSentTo')}</span>
                                <span className="ml-1 text-sm font-semibold text-[color:var(--tx)]">{phone}</span>
                                <button
                                    type="button"
                                    onClick={() => { setStep('phone'); setError('') }}
                                    className="ml-2 text-xs text-[color:var(--ac)] hover:opacity-70"
                                >
                                    {t('authBackToPhone')}
                                </button>
                            </div>

                            <InputField
                                type="text"
                                placeholder={t('authSmsCodePlaceholder')}
                                value={smsCode}
                                onChange={setSmsCode}
                                disabled={loading}
                                maxLength={6}
                                autoFocus
                            />

                            {/* Resend */}
                            <div className="text-right">
                                {countdown > 0
                                    ? <span className="text-xs text-[color:var(--tx2)]">{t('authResendIn').replace('{n}', String(countdown))}</span>
                                    : (
                                        <button
                                            type="button"
                                            className="text-xs text-[color:var(--ac)] hover:opacity-70"
                                            onClick={() => { setStep('phone'); setError(''); void fetchCaptcha() }}
                                        >
                                            {t('authResend')}
                                        </button>
                                    )
                                }
                            </div>
                        </>
                    )}

                    {/* Error */}
                    {error && (
                        <div className="rounded-xl bg-[rgba(220,53,69,0.08)] px-4 py-3 text-sm text-[#dc3545]">
                            {error}
                        </div>
                    )}

                    {/* Submit button */}
                    <button
                        type="submit"
                        disabled={loading}
                        className="login-page__submit"
                    >
                        {loading
                            ? <><Spinner /><span>{t('authLoggingIn')}</span></>
                            : mode === 'sms' && step === 'phone'
                                ? t('authSendSms')
                                : t('authLoginBtn')
                        }
                    </button>
                </form>

                {/* Footer note */}
                <div className="login-page__footer-note">
                    {t('authFooterNote')}
                </div>

                {/* 用户协议勾选 */}
                <div className="mt-3 flex items-start gap-2 px-1">
                    <input
                        id="agree-terms"
                        type="checkbox"
                        checked={agreed}
                        onChange={(e) => setAgreed(e.target.checked)}
                        className="mt-0.5 h-3.5 w-3.5 cursor-pointer accent-[color:var(--ac)]"
                    />
                    <label htmlFor="agree-terms" className="text-xs leading-relaxed text-[color:var(--tx2)] cursor-pointer select-none">
                        我已阅读并同意{' '}
                        <button
                            type="button"
                            onClick={() => setShowTerms(true)}
                            className="text-[color:var(--ac)] hover:opacity-70 font-medium"
                        >
                            {t('authTermsLink')}
                        </button>
                    </label>
                </div>
            </div>

            {/* 用户协议弹窗 */}
            {showTerms && (
                <div
                    className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 px-4"
                    onClick={() => setShowTerms(false)}
                >
                    <div
                        className="glass-panel flex flex-col w-full max-w-lg max-h-[80vh] rounded-2xl shadow-2xl overflow-hidden"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="flex items-center justify-between px-6 py-4 border-b border-[color:var(--bd)]">
                            <h2 className="text-base font-bold text-[color:var(--tx)]">{t('authTermsTitle')}</h2>
                            <button
                                type="button"
                                onClick={() => setShowTerms(false)}
                                className="p-1.5 rounded-lg hover:bg-[color:var(--bg2)] text-[color:var(--tx2)]"
                            >
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                                </svg>
                            </button>
                        </div>
                        <div className="overflow-y-auto px-6 py-4 text-xs text-[color:var(--tx2)] leading-relaxed space-y-4">
                            <TermsContent />
                        </div>
                        <div className="px-6 py-4 border-t border-[color:var(--bd)]">
                            <button
                                type="button"
                                onClick={() => { setAgreed(true); setShowTerms(false) }}
                                className="login-page__submit"
                            >
                                {t('authTermsAgreeBtn')}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    )
}

// ─── 协议内容 ─────────────────────────────────────────────────────────────────

function TermsContent() {
    return (
        <>
            <p className="text-[color:var(--tx2)] opacity-70">更新日期：2026年4月23日 &nbsp;|&nbsp; 生效日期：2026年4月23日</p>

            <section>
                <h3 className="font-semibold text-[color:var(--tx)] mb-1">一、总则</h3>
                <p>EvoMind 平台（以下简称"本平台"）由上海物智进化科技有限公司（Evomind-tech，以下简称"我们"）开发和运营。本协议适用于您使用本平台的全部服务。请您在注册或使用本平台前仔细阅读本协议，使用本平台即表示您已充分理解并同意本协议的全部条款。</p>
            </section>

            <section>
                <h3 className="font-semibold text-[color:var(--tx)] mb-1">二、账号注册与安全</h3>
                <p>1. 您需要提供真实有效的手机号码进行注册，并对账号安全负责。</p>
                <p>2. 您不得将账号转让、出售或授权他人使用。</p>
                <p>3. 若发现账号被盗用或存在安全风险，请立即联系我们。</p>
            </section>

            <section>
                <h3 className="font-semibold text-[color:var(--tx)] mb-1">三、数据上传与使用规范</h3>
                <p>1. 您上传的机器人操作数据集须为您合法拥有或经授权的数据，不得包含侵犯第三方权益的内容。</p>
                <p>2. 上传数据须符合 LeRobot 格式规范，不得上传恶意代码、病毒或其他有害内容。</p>
                <p>3. 您上传的公开数据集默认遵循您选择的开源许可证，我们不对数据内容的准确性作担保。</p>
            </section>

            <section>
                <h3 className="font-semibold text-[color:var(--tx)] mb-1">四、隐私政策</h3>
                <p>1. <strong>收集的信息</strong>：手机号码（用于身份验证）、上传的数据集文件及元信息、操作日志。</p>
                <p>2. <strong>使用目的</strong>：提供平台服务、保障账号安全、改善用户体验。</p>
                <p>3. <strong>数据存储</strong>：您的数据存储在阿里云（中国大陆）的服务器上，我们采取合理的技术措施保护数据安全。</p>
                <p>4. <strong>数据共享</strong>：我们不会将您的个人信息出售给第三方。在法律要求或保护我们合法权益时，我们可能依法披露相关信息。</p>
                <p>5. <strong>数据删除</strong>：您可联系我们申请删除您的账号及相关数据。</p>
            </section>

            <section>
                <h3 className="font-semibold text-[color:var(--tx)] mb-1">五、免责声明</h3>
                <p>1. 本平台提供的数据集及相关内容由用户上传，我们不对其准确性、完整性或适用性作任何保证。</p>
                <p>2. 因不可抗力、第三方服务故障等导致的服务中断，我们不承担相应责任。</p>
                <p>3. 您因使用平台数据而产生的任何后果，由您自行承担。</p>
            </section>

            <section>
                <h3 className="font-semibold text-[color:var(--tx)] mb-1">六、协议变更</h3>
                <p>我们保留随时修改本协议的权利。重大变更将通过平台公告或短信通知。继续使用本平台即视为接受修改后的协议。</p>
            </section>

            <section>
                <h3 className="font-semibold text-[color:var(--tx)] mb-1">七、联系我们</h3>
                <p>上海物智进化科技有限公司（Evomind-tech）</p>
                <p>联系邮箱：contact@evomind-tech.com</p>
            </section>
        </>
    )
}
