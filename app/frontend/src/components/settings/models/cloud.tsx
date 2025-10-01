import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import type { ProviderCapabilities } from '@/services/types';
import { Cloud, RefreshCw } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';

interface CloudModelsProps {
  className?: string;
}

interface CloudModel {
  display_name: string;
  model_name: string;
}

interface ProviderGroup {
  name: string;
  models: CloudModel[];
  capabilities?: ProviderCapabilities;
  api_key_env?: string[];
}

export function CloudModels({ className }: CloudModelsProps) {
  const [providers, setProviders] = useState<ProviderGroup[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchProviders = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/language-models/providers');
      if (response.ok) {
        const data = await response.json();
        setProviders(data.providers);
      } else {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        setError(`Failed to fetch providers: ${errorData.detail}`);
      }
    } catch (error) {
      console.error('Failed to fetch cloud model providers:', error);
      setError('Failed to connect to backend service');
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchProviders();
  }, []);

  const sortedProviders = useMemo(
    () => [...providers].sort((a, b) => a.name.localeCompare(b.name)),
    [providers]
  );

  const totalModels = useMemo(
    () => providers.reduce((count, provider) => count + provider.models.length, 0),
    [providers]
  );

  const renderCapabilityBadges = (capabilities?: ProviderCapabilities) => {
    if (!capabilities) {
      return null;
    }

    const badges: Array<{ key: string; label: string; className: string }> = [];

    if (capabilities.supports_reasoning) {
      badges.push({
        key: 'reasoning',
        label: 'Reasoning mode',
        className: 'border-purple-500/40 bg-purple-500/10 text-purple-200',
      });
    }

    if (capabilities.supports_json_mode === true) {
      badges.push({
        key: 'json-mode',
        label: 'JSON mode',
        className: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-200',
      });
    }

    if (capabilities.supports_json_mode === false) {
      badges.push({
        key: 'no-json-mode',
        label: 'No JSON mode',
        className: 'border-amber-500/40 bg-amber-500/10 text-amber-200',
      });
    }

    if (badges.length === 0) {
      return null;
    }

    return (
      <div className="flex flex-wrap gap-2">
        {badges.map(badge => (
          <Badge
            key={badge.key}
            variant="outline"
            className={cn('text-xs font-medium', badge.className)}
          >
            {badge.label}
          </Badge>
        ))}
      </div>
    );
  };

  return (
    <div className={cn("space-y-6", className)}>

      {error && (
        <div className="bg-red-900/20 border border-red-600/30 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <Cloud className="h-5 w-5 text-red-500 mt-0.5" />
            <div>
              <h4 className="font-medium text-red-300">Error</h4>
              <p className="text-sm text-red-500 mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}

      <div className="space-y-2">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-medium text-primary
          ">Available Models</h3>
          <span className="text-xs text-muted-foreground">
            {totalModels} models from {providers.length} providers
          </span>
        </div>

        {loading ? (
          <div className="text-center py-8">
            <RefreshCw className="h-8 w-8 mx-auto mb-2 animate-spin text-muted-foreground" />
            <p className="text-sm text-muted-foreground">Loading cloud models...</p>
          </div>
        ) : sortedProviders.length > 0 ? (
          <div className="space-y-4">
            {sortedProviders.map(provider => (
              <div
                key={provider.name}
                className="rounded-lg border border-border/40 bg-muted/40 p-4 space-y-3"
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <h4 className="font-semibold text-primary">{provider.name}</h4>
                    {provider.api_key_env?.length ? (
                      <p className="text-xs text-muted-foreground mt-1">
                        API key: {provider.api_key_env.join(' or ')}
                      </p>
                    ) : null}
                  </div>
                  {renderCapabilityBadges(provider.capabilities)}
                </div>

                {provider.capabilities?.notes?.length ? (
                  <ul className="ml-1 list-disc space-y-1 pl-4 text-xs text-muted-foreground">
                    {provider.capabilities.notes.map(note => (
                      <li key={note}>{note}</li>
                    ))}
                  </ul>
                ) : null}

                <div className="space-y-1">
                  {provider.models.map(model => (
                    <div
                      key={`${provider.name}-${model.model_name}`}
                      className="group flex items-center justify-between rounded-md bg-background/40 px-3 py-2.5 transition-colors hover:bg-background/60"
                    >
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-sm truncate text-primary">{model.display_name}</span>
                          {model.model_name !== model.display_name && (
                            <span className="font-mono text-xs text-muted-foreground">
                              {model.model_name}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        ) : (
          !loading && (
            <div className="text-center py-8 text-muted-foreground">
              <Cloud className="h-8 w-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No models available</p>
            </div>
          )
        )}
      </div>
    </div>
  );
} 