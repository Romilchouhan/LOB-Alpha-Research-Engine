// ─────────────────────────────────────────────────────────────────────────────
// Hazelcast distributed store — Implementation
// ─────────────────────────────────────────────────────────────────────────────

#include "infra/hazelcast_store.hpp"
#include <iostream>

#if defined(LOB_HAS_HAZELCAST) && LOB_HAS_HAZELCAST
#include <hazelcast/client/hazelcast_client.h>
#endif

namespace infra {

// ── No-op stub (used when Hazelcast is not linked) ──────────────────────────

class NoOpStore final : public IDistributedStore {
public:
    bool connect(const std::string& /*addr*/) override {
        std::cout << "[Store] Hazelcast DISABLED — running in local-only mode\n";
        return false;
    }

    void disconnect() override {}

    void put_trade_signal(const std::string& /*key*/,
                          const std::string& /*json*/) override {}

    void put_inventory_risk(const std::string& /*key*/,
                            double /*risk*/) override {}
};

// ── Hazelcast-backed store ──────────────────────────────────────────────────

#if defined(LOB_HAS_HAZELCAST) && LOB_HAS_HAZELCAST

class HazelcastStore final : public IDistributedStore {
public:
    bool connect(const std::string& addr) override {
        try {
            hazelcast::client::client_config cfg;
            cfg.get_network_config().add_address({addr, 5701});
            // Create client and move it into unique_ptr
            client_ = std::make_unique<hazelcast::client::hazelcast_client>(
                hazelcast::new_client(std::move(cfg)).get()
            );
            signals_map_   = client_->get_map("Trade_Signals").get();
            inventory_map_ = client_->get_map("Inventory_Risk").get();
            std::cout << "[Store] Connected to Hazelcast cluster @ " << addr << "\n";
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[Store] Hazelcast connection failed: " << e.what() << "\n";
            return false;
        }
    }

    void disconnect() override {
        if (client_) {
            client_->shutdown().get();
        }
    }

    void put_trade_signal(const std::string& key,
                          const std::string& json) override {
        if (signals_map_) {
            signals_map_->put<std::string, std::string>(key, json).get();
        }
    }

    void put_inventory_risk(const std::string& key,
                            double risk) override {
        if (inventory_map_) {
            inventory_map_->put<std::string, double>(key, risk).get();
        }
    }

private:
    std::unique_ptr<hazelcast::client::hazelcast_client> client_;
    std::shared_ptr<hazelcast::client::imap> signals_map_;
    std::shared_ptr<hazelcast::client::imap> inventory_map_;
};

#endif  // LOB_HAS_HAZELCAST

// ── Factory ─────────────────────────────────────────────────────────────────

std::unique_ptr<IDistributedStore> make_store(bool enable_hazelcast) {
#if defined(LOB_HAS_HAZELCAST) && LOB_HAS_HAZELCAST
    if (enable_hazelcast)
        return std::make_unique<HazelcastStore>();
#else
    (void)enable_hazelcast;
#endif
    return std::make_unique<NoOpStore>();
}

} // namespace infra
