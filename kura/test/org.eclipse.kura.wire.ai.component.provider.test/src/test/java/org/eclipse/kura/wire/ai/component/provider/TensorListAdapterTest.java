/*******************************************************************************
 * Copyright (c) 2022 Eurotech and/or its affiliates and others
 *
 * This program and the accompanying materials are made
 * available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 *
 * SPDX-License-Identifier: EPL-2.0
 *
 * Contributors:
 *  Eurotech
 ******************************************************************************/

package org.eclipse.kura.wire.ai.component.provider;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.fail;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.eclipse.kura.KuraIOException;
import org.eclipse.kura.ai.inference.Tensor;
import org.eclipse.kura.ai.inference.TensorDescriptor;
import org.eclipse.kura.type.FloatValue;
import org.eclipse.kura.type.TypedValue;
import org.eclipse.kura.wire.WireRecord;
import org.junit.Test;

public class TensorListAdapterTest {

    private TensorListAdapter adapterInstance = new TensorListAdapter();

    /*
     * Scenarios
     */
    @Test
    public void initialTest() {
        // Build WireRecord
        Map<String, TypedValue<?>> wireRecordProperties = new HashMap();
        wireRecordProperties.put("INPUT0", new FloatValue(1.0F));

        WireRecord inputRecord = new WireRecord(wireRecordProperties);

        // Build TensorDescriptor
        String name = "INPUT0";
        String type = "FP32";
        Optional<String> format = Optional.empty();
        List<Long> shape = Arrays.asList(1L, 1L);
        Map<String, Object> parameters = new HashMap<>();

        TensorDescriptor descriptor = new TensorDescriptor(name, type, format, shape, parameters);

        List<TensorDescriptor> descriptorList = Arrays.asList(descriptor);

        // Initialize TensorListAdapter
        TensorListAdapter.givenDescriptors(descriptorList);

        // Attempt conversion from wire records
        try {
            List<Tensor> result = adapterInstance.fromWireRecord(inputRecord);

            assertFalse(result.isEmpty());
            assertEquals(1, result.size());

            Tensor resultingTensor = result.get(0);

            assertEquals(Float.class, resultingTensor.getType());
        } catch (KuraIOException e) {
            e.printStackTrace();
            fail("Unexpected exception was thrown");
        }

    }

    /*
     * Given
     */

    /*
     * When
     */

    /*
     * Then
     */

}
