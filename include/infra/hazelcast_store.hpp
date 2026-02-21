#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Hazelcast distributed store — Trade_Signals & Inventory_Risk maps
// ─────────────────────────────────────────────────────────────────────────────

#include <string>
#include <memory>

namespace infra {

/// Abstract interface so the main loop doesn't depend on Hazelcast headers.
class IDistributedStore {
public:
    virtual ~IDistributedStore() = default;

    virtual bool connect(const std::string& cluster_address = "127.0.0.1:5701") = 0;
    virtual void disconnect() = 0;

    virtual void put_trade_signal(const std::string& key,
                                  const std::string& json_value) = 0;

    virtual void put_inventory_risk(const std::string& key,
                                    double risk_value) = 0;
};

/// Factory: returns a Hazelcast-backed store if LOB_HAS_HAZELCAST, else a no-op stub.
std::unique_ptr<IDistributedStore> make_store(bool enable_hazelcast);

} // namespace infra
