import unittest

from server import app


class GalaxyApiTest(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        response = self.client.post('/api/galaxy', json={'n_stars': 200, 't_max': 20.0})
        self.assertEqual(response.status_code, 200)
        self.payload = response.get_json()
        self.cache_id = self.payload['cache_id']

    def test_galaxy_payload_contains_star_flags(self):
        stars = self.payload['stars']
        self.assertIn('stellar_model', self.payload)
        self.assertEqual(self.payload['stellar_model'], 'auto')
        self.assertIn('met', stars)
        self.assertIn('has_planets', stars)
        self.assertIn('has_hz', stars)
        self.assertIn('has_moon_system', stars)
        self.assertIn('has_regular_moons', stars)
        self.assertIn('has_irregular_moons', stars)
        self.assertIn('has_large_moon', stars)
        self.assertIn('has_resonant_moon_chain', stars)
        self.assertIn('moon_count_estimate', stars)
        self.assertEqual(len(stars['met']), 200)
        self.assertEqual(len(stars['has_planets']), 200)
        self.assertEqual(len(stars['has_hz']), 200)
        self.assertEqual(len(stars['has_moon_system']), 200)
        self.assertEqual(len(stars['moon_count_estimate']), 200)

    def test_star_and_evolution_endpoints_accept_cache_id(self):
        star = self.client.get(f'/api/star/5?t=13.8&cache_id={self.cache_id}')
        self.assertEqual(star.status_code, 200)
        star_payload = star.get_json()
        self.assertEqual(star_payload['cache_id'], self.cache_id)
        self.assertEqual(star_payload['stellar_model'], 'auto')
        self.assertIn('planets', star_payload)
        self.assertIn('evolution', star_payload)

        evo = self.client.get(f'/api/evolution/5?t_max=100.0&cache_id={self.cache_id}')
        self.assertEqual(evo.status_code, 200)
        evo_payload = evo.get_json()
        self.assertEqual(evo_payload['cache_id'], self.cache_id)
        self.assertEqual(evo_payload['stellar_model'], 'auto')
        self.assertGreater(len(evo_payload['track']), 0)

    def test_stellar_model_can_be_selected_and_overridden(self):
        defaults = self.client.get('/api/defaults')
        self.assertEqual(defaults.status_code, 200)
        defaults_payload = defaults.get_json()
        self.assertEqual(defaults_payload['stellar_model'], 'auto')
        self.assertIn('precise', defaults_payload['stellar_model_options'])

        precise = self.client.post('/api/galaxy', json={'n_stars': 60, 't_max': 20.0, 'stellar_model': 'precise'})
        self.assertEqual(precise.status_code, 200)
        precise_payload = precise.get_json()
        self.assertEqual(precise_payload['stellar_model'], 'precise')
        precise_cache = precise_payload['cache_id']

        star = self.client.get(f'/api/star/5?t=13.8&cache_id={precise_cache}')
        self.assertEqual(star.status_code, 200)
        self.assertEqual(star.get_json()['stellar_model'], 'precise')

        evo = self.client.get(f'/api/evolution/5?t_max=50.0&stellar_model=heuristic&cache_id={precise_cache}')
        self.assertEqual(evo.status_code, 200)
        self.assertEqual(evo.get_json()['stellar_model'], 'heuristic')

    def test_star_payload_exposes_moon_system_for_gas_giants(self):
        stars = self.payload['stars']
        host_ids = [idx for idx, flag in enumerate(stars.get('has_gas', [])) if flag]
        self.assertTrue(host_ids, 'Expected at least one gas-giant host in deterministic test galaxy')

        for star_id in host_ids[:40]:
            response = self.client.get(f'/api/star/{star_id}?t=13.8&cache_id={self.cache_id}')
            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            for planet in payload.get('planets', []):
                if planet.get('type') != 'gas_giant' or planet.get('status') != 'active':
                    continue
                self.assertIn('moon_system', planet)
                self.assertIn('has_moon_system', planet)
                self.assertIn('moon_count', planet)
                self.assertIsInstance(planet['moon_system'], dict)
                self.assertIn('cpd', planet['moon_system'])
                self.assertIn('summary', planet['moon_system'])
                self.assertIn('regular_moons', planet['moon_system'])
                self.assertIn('irregular_moons', planet['moon_system'])
                return

        self.fail('Could not find an active gas giant in the API response to validate moon fields')

    def test_star_payload_exposes_moon_system_for_neptune_class_hosts(self):
        for star_id in range(min(80, len(self.payload['stars']['mass']))):
            response = self.client.get(f'/api/star/{star_id}?t=13.8&cache_id={self.cache_id}')
            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            for planet in payload.get('planets', []):
                if planet.get('type') != 'mini_neptune' or planet.get('status') != 'active':
                    continue
                self.assertIn('moon_system', planet)
                self.assertIn('has_moon_system', planet)
                self.assertIn('moon_count', planet)
                self.assertIsInstance(planet['moon_system'], dict)
                self.assertIn('cpd', planet['moon_system'])
                self.assertIn('summary', planet['moon_system'])
                self.assertIn('regular_moons', planet['moon_system'])
                self.assertIn('irregular_moons', planet['moon_system'])
                self.assertEqual(planet['moon_system']['cpd']['host_regime'], 'neptunian')
                return

        self.fail('Could not find an active Neptune-class planet in the API response to validate moon fields')

    def test_star_payload_exposes_moon_system_for_rocky_hosts(self):
        for star_id in range(min(80, len(self.payload['stars']['mass']))):
            response = self.client.get(f'/api/star/{star_id}?t=13.8&cache_id={self.cache_id}')
            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            for planet in payload.get('planets', []):
                if planet.get('type') not in ('rocky', 'hot_rocky') or planet.get('status') != 'active':
                    continue
                self.assertIn('moon_system', planet)
                self.assertIn('has_moon_system', planet)
                self.assertIn('moon_count', planet)
                self.assertIsInstance(planet['moon_system'], dict)
                self.assertIn('formation_channel', planet['moon_system'])
                self.assertIn('impact_state', planet['moon_system'])
                self.assertIn('debris_disk', planet['moon_system'])
                self.assertIn('summary', planet['moon_system'])
                self.assertIn('major_moons', planet['moon_system'])
                self.assertIn('minor_moons', planet['moon_system'])
                return

        self.fail('Could not find an active rocky planet in the API response to validate moon fields')

    def test_invalid_inputs_return_client_errors(self):
        bad_star = self.client.get(f'/api/star/5?t=nan&cache_id={self.cache_id}')
        self.assertEqual(bad_star.status_code, 400)

        bad_evo = self.client.get(f'/api/evolution/999999?t_max=100.0&cache_id={self.cache_id}')
        self.assertEqual(bad_evo.status_code, 404)

        too_many = self.client.post('/api/galaxy', json={'n_stars': 10**9, 't_max': 20.0})
        self.assertEqual(too_many.status_code, 400)

        bad_tmax = self.client.get(f'/api/evolution/5?t_max=inf&cache_id={self.cache_id}')
        self.assertEqual(bad_tmax.status_code, 400)

        bad_model = self.client.post('/api/galaxy', json={'n_stars': 20, 't_max': 20.0, 'stellar_model': 'invalid'})
        self.assertEqual(bad_model.status_code, 400)


if __name__ == '__main__':
    unittest.main()
