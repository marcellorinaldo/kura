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
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.eclipse.kura.KuraIOException;
import org.eclipse.kura.ai.inference.Tensor;
import org.eclipse.kura.ai.inference.TensorDescriptor;
import org.eclipse.kura.type.BooleanValue;
import org.eclipse.kura.type.DoubleValue;
import org.eclipse.kura.type.FloatValue;
import org.eclipse.kura.type.IntegerValue;
import org.eclipse.kura.type.LongValue;
import org.eclipse.kura.type.TypedValue;
import org.eclipse.kura.wire.WireRecord;
import org.junit.Before;
import org.junit.Test;

public class TensorListAdapterTest {

    private TensorListAdapter adapterInstance = new TensorListAdapter();

    private Map<String, TypedValue<?>> wireRecordProperties;
    private WireRecord inputRecord;

    private List<TensorDescriptor> inputDescriptors;

    private List<Tensor> outputTensors;

    private boolean exceptionOccurred = false;

    /*
     * Scenarios
     */
    @Test
    public void adapterShouldWorkWithBooleanScalar() {
        givenWireRecordPropWith("INPUT0", new BooleanValue(true));
        givenWireRecord();

        givenScalarTensorDescriptorWith("INPUT0", "BOOL");
        givenDescriptorToTensorListAdapter();

        whenTensorListAdapterConvertsFromWireRecord();

        thenNoExceptionOccurred();
        thenResultingScalarTensorIsIstanceOf(Boolean.class);
        thenResultingScalarTensorIsEqualTo(Boolean.class, new Boolean(true));
    }

    @Test
    public void adapterShouldWorkWithByteArray() {
        // TODO
        assertTrue(true);
    }

    @Test
    public void adapterShouldWorkWithFloatScalar() {
        givenWireRecordPropWith("INPUT0", new FloatValue(1.0F));
        givenWireRecord();

        givenScalarTensorDescriptorWith("INPUT0", "FP32");
        givenDescriptorToTensorListAdapter();

        whenTensorListAdapterConvertsFromWireRecord();

        thenNoExceptionOccurred();
        thenResultingScalarTensorIsIstanceOf(Float.class);
        thenResultingScalarTensorIsEqualTo(Float.class, new Float(1.0F));
    }

    @Test
    public void adapterShouldWorkWithDoubleScalar() {
        givenWireRecordPropWith("INPUT0", new DoubleValue(3.0F));
        givenWireRecord();

        givenScalarTensorDescriptorWith("INPUT0", "FP32");
        givenDescriptorToTensorListAdapter();

        whenTensorListAdapterConvertsFromWireRecord();

        thenNoExceptionOccurred();
        thenResultingScalarTensorIsIstanceOf(Double.class);
        thenResultingScalarTensorIsEqualTo(Double.class, new Double(3.0F));
    }

    @Test
    public void adapterShouldWorkWithIntegerScalar() {
        givenWireRecordPropWith("INPUT0", new IntegerValue(6));
        givenWireRecord();

        givenScalarTensorDescriptorWith("INPUT0", "INT32");
        givenDescriptorToTensorListAdapter();

        whenTensorListAdapterConvertsFromWireRecord();

        thenNoExceptionOccurred();
        thenResultingScalarTensorIsIstanceOf(Integer.class);
        thenResultingScalarTensorIsEqualTo(Integer.class, new Integer(6));
    }

    @Test
    public void adapterShouldWorkWithLongScalar() {
        givenWireRecordPropWith("INPUT0", new LongValue(6555));
        givenWireRecord();

        givenScalarTensorDescriptorWith("INPUT0", "INT32");
        givenDescriptorToTensorListAdapter();

        whenTensorListAdapterConvertsFromWireRecord();

        thenNoExceptionOccurred();
        thenResultingScalarTensorIsIstanceOf(Long.class);
        thenResultingScalarTensorIsEqualTo(Long.class, new Long(6555));
    }

    @Test
    public void adapterShouldWorkWithStringScalar() {
        // TODO
        assertTrue(true);
    }

    @Test
    public void adapterShouldWorkWithMultipleFloat() {
        // Build WireRecord
        Map<String, TypedValue<?>> wireRecordProperties = new HashMap();
        wireRecordProperties.put("INPUT0", new FloatValue(1.0F));
        wireRecordProperties.put("INPUT1", new FloatValue(1.0F));
        wireRecordProperties.put("INPUT2", new FloatValue(1.0F));
        wireRecordProperties.put("INPUT3", new FloatValue(1.0F));

        WireRecord inputRecord = new WireRecord(wireRecordProperties);

        // Build TensorDescriptor
        String type = "FP32";
        Optional<String> format = Optional.empty();
        List<Long> shape = Arrays.asList(1L, 1L);
        Map<String, Object> parameters = new HashMap<>();

        TensorDescriptor descriptor_0 = new TensorDescriptor("INPUT0", type, format, shape, parameters);
        TensorDescriptor descriptor_1 = new TensorDescriptor("INPUT1", type, format, shape, parameters);
        TensorDescriptor descriptor_2 = new TensorDescriptor("INPUT2", type, format, shape, parameters);
        TensorDescriptor descriptor_3 = new TensorDescriptor("INPUT3", type, format, shape, parameters);

        List<TensorDescriptor> descriptorList = Arrays.asList(descriptor_0, descriptor_1, descriptor_2, descriptor_3);

        // Initialize TensorListAdapter
        TensorListAdapter.givenDescriptors(descriptorList);

        // Attempt conversion from wire records
        try {
            List<Tensor> result = adapterInstance.fromWireRecord(inputRecord);

            assertFalse(result.isEmpty());
            assertEquals(4, result.size());

            for (Tensor resultingTensor : result) {
                assertEquals(Float.class, resultingTensor.getType());

                Optional<List<Float>> data = resultingTensor.getData(Float.class);

                assertTrue(data.isPresent());
                assertEquals(new Float(1.0F), data.get().get(0));
            }
        } catch (KuraIOException e) {
            e.printStackTrace();
            fail("Unexpected exception was thrown");
        }
    }

    /*
     * Given
     */
    private void givenWireRecordPropWith(String name, TypedValue<?> value) {
        this.wireRecordProperties.put(name, value);
    }

    private void givenWireRecord() {
        this.inputRecord = new WireRecord(this.wireRecordProperties);
    }

    private void givenScalarTensorDescriptorWith(String name, String type) {
        Optional<String> format = Optional.empty();
        List<Long> shape = Arrays.asList(1L, 1L);
        Map<String, Object> parameters = new HashMap<>();

        TensorDescriptor descriptor = new TensorDescriptor(name, type, format, shape, parameters);

        this.inputDescriptors = Arrays.asList(descriptor);
    }

    private void givenDescriptorToTensorListAdapter() {
        TensorListAdapter.givenDescriptors(this.inputDescriptors);
    }

    /*
     * When
     */
    private void whenTensorListAdapterConvertsFromWireRecord() {
        try {
            this.outputTensors = adapterInstance.fromWireRecord(inputRecord);
        } catch (KuraIOException e) {
            e.printStackTrace();
            this.exceptionOccurred = true;
        }
    }

    /*
     * Then
     */
    private void thenNoExceptionOccurred() {
        assertFalse(this.exceptionOccurred);
    }

    private <T> void thenResultingScalarTensorIsIstanceOf(Class<T> type) {
        assertEquals(1, this.outputTensors.size());
        Optional<List<T>> data = this.outputTensors.get(0).getData(type);

        assertTrue(data.isPresent());
        assertEquals(1, data.get().size());
    }

    private <T> void thenResultingScalarTensorIsEqualTo(Class<T> type, T value) {
        assertEquals(1, this.outputTensors.size());
        Optional<List<T>> data = this.outputTensors.get(0).getData(type);

        assertTrue(data.isPresent());
        assertEquals(value, data.get().get(0));
    }

    /*
     * Utils
     */
    @Before
    public void inputWireRecordPropCleanup() {
        this.wireRecordProperties = new HashMap();
    }

}
