import type { ProviderOption } from '@/domains/provider/api/providerApi'

export function providerDisplayLabel(
  provider: Pick<ProviderOption, 'name' | 'label'> | null | undefined,
  customLabel: string,
): string {
  if (!provider) return ''
  return provider.name === 'custom' ? customLabel : provider.label
}
