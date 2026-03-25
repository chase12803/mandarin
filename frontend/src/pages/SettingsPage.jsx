import React, { useState, useEffect } from "react";
import { getSettings, putSettings, getModels } from "../api/client";

const PROVIDER_LABELS = {
  openai: "OpenAI (GPT)",
  anthropic: "Anthropic (Claude)",
  google: "Google (Gemini)",
  tavily: "Tavily (Web Search)",
};
const PROVIDERS = ["openai", "anthropic", "google", "tavily"];
const BLANK_API_KEY_STATE = PROVIDERS.reduce((acc, provider) => {
  acc[provider] = "";
  return acc;
}, {});
const HIDDEN_KEY_INPUT_STATE = PROVIDERS.reduce((acc, provider) => {
  acc[provider] = false;
  return acc;
}, {});
const CLEAR_ALL_KEYS_HELP_TEXT =
  "Clears all API keys saved in app settings after confirmation. This helps remove local sensitive data for security/data handling reasons.";
const WEB_SEARCH_MODE_OPTIONS = [
  { id: "off", label: "Off" },
  { id: "native", label: "Native" },
  { id: "tavily", label: "Tavily" },
];

export default function SettingsPage() {
  const [settings, setSettings] = useState(null);
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [defaultModel, setDefaultModel] = useState("");
  const [defaultWebSearchMode, setDefaultWebSearchMode] = useState("off");
  const [aiMemoryEnabled, setAiMemoryEnabled] = useState(false);
  const [apiKeyDrafts, setApiKeyDrafts] = useState(() => ({ ...BLANK_API_KEY_STATE }));
  const [showKeyInput, setShowKeyInput] = useState(() => ({ ...HIDDEN_KEY_INPUT_STATE }));
  const hasAnyApiKeySet = PROVIDERS.some((provider) => settings?.api_keys?.[provider]?.set);

  useEffect(() => {
    Promise.all([getSettings(), getModels()])
      .then(([settingsData, modelsData]) => {
        setSettings(settingsData);
        const available = (modelsData || []).filter((m) => m.available);
        setModels(available);
        setDefaultModel(settingsData.default_model || "");
        setDefaultWebSearchMode((settingsData.default_web_search_mode || "off").toLowerCase());
        setAiMemoryEnabled(settingsData.ai_memory_enabled === true);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const handleSaveDefaultModel = () => {
    setSaving(true);
    setError(null);
    setSuccess(null);
    putSettings({
      default_model: defaultModel || null,
      default_web_search_mode: defaultWebSearchMode,
    })
      .then((data) => {
        setSettings(data);
        setDefaultWebSearchMode((data.default_web_search_mode || "off").toLowerCase());
        setAiMemoryEnabled(data.ai_memory_enabled === true);
        setSuccess("Default model and web search saved.");
        setTimeout(() => setSuccess(null), 3000);
      })
      .catch((e) => setError(e.message))
      .finally(() => setSaving(false));
  };

  const handleToggleAiMemory = () => {
    const next = !aiMemoryEnabled;
    setSaving(true);
    setError(null);
    setSuccess(null);
    putSettings({ ai_memory_enabled: next })
      .then((data) => {
        setSettings(data);
        setAiMemoryEnabled(data.ai_memory_enabled === true);
        setSuccess(`AI-managed memory ${data.ai_memory_enabled ? "enabled" : "disabled"}.`);
        setTimeout(() => setSuccess(null), 3000);
      })
      .catch((e) => setError(e.message))
      .finally(() => setSaving(false));
  };

  const handleSaveApiKey = (provider) => {
    const value = (apiKeyDrafts[provider] || "").trim();
    setSaving(true);
    setError(null);
    setSuccess(null);
    putSettings({
      api_keys: { [provider]: value },
    })
      .then((data) => {
        setSettings(data);
        setApiKeyDrafts((prev) => ({ ...prev, [provider]: "" }));
        setShowKeyInput((prev) => ({ ...prev, [provider]: false }));
        setSuccess(`${PROVIDER_LABELS[provider]} API key ${value ? "saved" : "removed"}.`);
        setTimeout(() => setSuccess(null), 3000);
      })
      .catch((e) => setError(e.message))
      .finally(() => setSaving(false));
  };

  const handleRemoveApiKey = (provider) => {
    if (!window.confirm(`Remove ${PROVIDER_LABELS[provider]} API key?`)) return;
    setSaving(true);
    setError(null);
    setSuccess(null);
    putSettings({ api_keys: { [provider]: "" } })
      .then((data) => {
        setSettings(data);
        setApiKeyDrafts((prev) => ({ ...prev, [provider]: "" }));
        setShowKeyInput((prev) => ({ ...prev, [provider]: false }));
        setSuccess(`${PROVIDER_LABELS[provider]} API key removed.`);
        setTimeout(() => setSuccess(null), 3000);
      })
      .catch((e) => setError(e.message))
      .finally(() => setSaving(false));
  };

  const handleClearAllApiKeys = () => {
    const confirmed = window.confirm(
      "Clear all API keys saved in app settings?\n\nThis removes locally stored keys for OpenAI, Anthropic, Google, and Tavily. Keys set in .env are not removed here."
    );
    if (!confirmed) return;
    setSaving(true);
    setError(null);
    setSuccess(null);
    const clearedApiKeys = PROVIDERS.reduce((acc, provider) => {
      acc[provider] = "";
      return acc;
    }, {});
    putSettings({ api_keys: clearedApiKeys })
      .then((data) => {
        setSettings(data);
        setApiKeyDrafts({ ...BLANK_API_KEY_STATE });
        setShowKeyInput({ ...HIDDEN_KEY_INPUT_STATE });
        setSuccess("All saved API keys cleared.");
        setTimeout(() => setSuccess(null), 3000);
      })
      .catch((e) => setError(e.message))
      .finally(() => setSaving(false));
  };

  const revealKeyInput = (provider) => {
    setShowKeyInput((prev) => ({ ...prev, [provider]: true }));
    setApiKeyDrafts((prev) => ({ ...prev, [provider]: "" }));
  };

  if (loading) {
    return (
      <div className="settings-page">
        <div className="settings-loading">Loading settings…</div>
      </div>
    );
  }

  return (
    <div className="settings-page">
      <header className="settings-header">
        <h1>Settings</h1>
        <p className="settings-subtitle">Manage your API keys and default model for new chats</p>
      </header>

      {error && (
        <div className="error-banner">
          {error}
          <button type="button" onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}
      {success && (
        <div className="settings-success">
          {success}
        </div>
      )}

      <section className="settings-section">
        <h2 className="settings-section-title">Models & API keys</h2>

        <div className="settings-card">
          <div className="settings-card-header">
            <h3>Default model</h3>
            <span className="settings-hint">Used when you start a new chat</span>
          </div>
          <div className="settings-card-body">
            <select
              className="settings-select"
              value={defaultModel}
              onChange={(e) => setDefaultModel(e.target.value)}
            >
              <option value="">First available</option>
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} ({m.provider})
                </option>
              ))}
            </select>
            <select
              className="settings-select"
              value={defaultWebSearchMode}
              onChange={(e) => setDefaultWebSearchMode((e.target.value || "off").toLowerCase())}
            >
              {WEB_SEARCH_MODE_OPTIONS.map((option) => (
                <option key={option.id} value={option.id}>
                  Default web search: {option.label}
                </option>
              ))}
            </select>
            {models.length === 0 && (
              <div className="settings-hint">
                No available models. Add provider API keys or verify provider connectivity.
              </div>
            )}
            <button
              type="button"
              className="settings-btn primary"
              onClick={handleSaveDefaultModel}
              disabled={saving}
            >
              {saving ? "Saving…" : "Save defaults"}
            </button>
          </div>
        </div>

        <div className="settings-card">
          <div className="settings-card-header">
            <h3>AI-managed memory</h3>
            <span className="settings-hint">
              Recall stored memories in the system prompt and save new facts after each reply. Off by default.
            </span>
          </div>
          <div className="settings-card-body settings-card-body-switch">
            <div className="settings-switch-row">
              <span id="ai-memory-switch-label" className="settings-switch-text">
                Enable in chat
              </span>
              <button
                type="button"
                id="ai-memory-switch"
                role="switch"
                aria-checked={aiMemoryEnabled}
                aria-labelledby="ai-memory-switch-label"
                disabled={saving}
                className={`settings-switch ${aiMemoryEnabled ? "settings-switch-on" : ""}`}
                onClick={handleToggleAiMemory}
              >
                <span className="settings-switch-thumb" aria-hidden />
              </button>
            </div>
          </div>
        </div>

        <div className="settings-card settings-api-keys">
          <div className="settings-card-header">
            <h3>API keys</h3>
            <span className="settings-hint">Stored locally. Keys from .env take precedence.</span>
            <div className="settings-api-key-actions">
              <button
                type="button"
                className="settings-btn danger small"
                onClick={handleClearAllApiKeys}
                disabled={saving || !hasAnyApiKeySet}
              >
                {saving ? "Saving…" : "Clear all API keys"}
              </button>
              <span className="settings-help-tooltip">
                <span
                  className="settings-help-icon"
                  tabIndex={0}
                  aria-describedby="clear-all-api-keys-help"
                >
                  ?
                </span>
                <span
                  id="clear-all-api-keys-help"
                  className="settings-help-tooltip-content"
                  role="tooltip"
                >
                  {CLEAR_ALL_KEYS_HELP_TEXT}
                </span>
              </span>
            </div>
          </div>
          <div className="settings-api-key-list">
            {PROVIDERS.map((provider) => {
              const keyInfo = settings?.api_keys?.[provider] || {};
              const isSet = keyInfo.set;
              const showInput = showKeyInput[provider];
              return (
                <div key={provider} className="settings-api-key-row">
                  <label className="settings-api-key-label">{PROVIDER_LABELS[provider]}</label>
                  {showInput || !isSet ? (
                    <div className="settings-api-key-input-row">
                      <input
                        type="password"
                        className="settings-input"
                        placeholder={isSet ? "Enter new key to replace" : "Paste your API key"}
                        value={apiKeyDrafts[provider]}
                        onChange={(e) =>
                          setApiKeyDrafts((prev) => ({ ...prev, [provider]: e.target.value }))
                        }
                        autoComplete="off"
                        autoFocus={showInput}
                      />
                      <button
                        type="button"
                        className="settings-btn primary small"
                        onClick={() => handleSaveApiKey(provider)}
                        disabled={saving}
                      >
                        Save
                      </button>
                      {isSet && (
                        <button
                          type="button"
                          className="settings-btn small"
                          onClick={() => {
                            setShowKeyInput((prev) => ({ ...prev, [provider]: false }));
                            setApiKeyDrafts((prev) => ({ ...prev, [provider]: "" }));
                          }}
                        >
                          Cancel
                        </button>
                      )}
                    </div>
                  ) : (
                    <div className="settings-api-key-masked-row">
                      <span className="settings-api-key-masked">{keyInfo.masked || "••••••••"}</span>
                      <button
                        type="button"
                        className="settings-btn-link"
                        onClick={() => revealKeyInput(provider)}
                      >
                        Update
                      </button>
                      <button
                        type="button"
                        className="settings-btn-link danger"
                        onClick={() => handleRemoveApiKey(provider)}
                      >
                        Remove
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </section>
    </div>
  );
}
